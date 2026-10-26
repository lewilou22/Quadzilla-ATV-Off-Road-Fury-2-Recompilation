#include "iop_emulator.h"
#include "imports/iop_cdvd.h"
#include "core/iop_cpu.h"
#include "imports/iop_heaplib.h"
#include "imports/iop_imports.h"
#include "imports/iop_intrman.h"
#include "imports/iop_ioman.h"
#include "core/iop_kernel.h"
#include "imports/iop_loadcore.h"
#include "core/iop_memory.h"
#include "services/iop_module_loader.h"
#include "services/iop_rpc.h"
#include "imports/iop_stdio.h"
#include "imports/iop_sysclib.h"
#include "imports/iop_sysmem.h"
#include "imports/iop_timrman.h"
#include "imports/iop_vblank.h"
#include "iop_emulator_const.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <utility>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kRamSize = IopMemory::RamSize;
        constexpr uint32_t kKernelHeapBase = IopMemory::HeapBase;
        constexpr uint32_t kKernelHeapLimit = IopMemory::HeapLimit;
        constexpr uint32_t kCallStackBase = kKernelHeapLimit;
        constexpr uint32_t kCallStackLimit = 0x001FFF00u;
        constexpr uint32_t kCallStackSize = 0x2000u;
        constexpr uint32_t kCallStackCapacity = (kCallStackLimit - kCallStackBase) / kCallStackSize;
        constexpr uint64_t kCdvdCompletionCycles = 128u;
        // Callback-driven sceCdRead completes after a disc-like delay
        // (~18000 sectors/s). Drivers pace their threads on it: 989SND's
        // stream transfer thread spin-waits on a DMA channel that only a
        // lower-priority thread frees, which works because a disc read is
        // much slower than the SPU2 DMA that precedes it.
        constexpr uint64_t kCdvdCyclesPerSector = 2048u;
        constexpr int kDmaSpu0CompletionIrq = 0x24;
        constexpr int kDmaSpu1CompletionIrq = 0x28;

        uint32_t physicalAddress(uint32_t address)
        {
            return IopMemory::physicalAddress(address);
        }

        int32_t sign16(uint32_t value)
        {
            return static_cast<int16_t>(value & 0xFFFFu);
        }

        bool iequals(std::string_view lhs, std::string_view rhs);

        // ATV_IOP_IMPORT_TRACE=lib1,lib2 logs every import call into those libraries.
        bool traceImport(std::string_view library)
        {
            static const std::string libraries = []
            {
                const char *value = std::getenv("ATV_IOP_IMPORT_TRACE");
                return value ? std::string(",") + value + "," : std::string();
            }();
            if (libraries.empty())
                return false;
            std::string key = ",";
            key.append(library);
            key.push_back(',');
            return libraries.find(key) != std::string::npos;
        }

        // ATV_IOP_TRACE_PC=low,high[,max] (hex addresses, decimal max) logs every
        // instruction executed in [low, high), with a few registers.
        struct TracePcRange
        {
            bool active = false;
            uint32_t low = 0u;
            uint32_t high = 0u;
            uint64_t limit = 20000u;
        };

        const TracePcRange &tracePcRange()
        {
            static const TracePcRange range = []
            {
                TracePcRange parsed;
                const char *value = std::getenv("ATV_IOP_TRACE_PC");
                if (!value || !*value)
                    return parsed;
                char *end = nullptr;
                parsed.low = static_cast<uint32_t>(std::strtoul(value, &end, 16));
                if (end && *end == ',')
                    parsed.high = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
                if (end && *end == ',')
                    parsed.limit = std::strtoull(end + 1, nullptr, 10);
                parsed.active = parsed.high > parsed.low;
                return parsed;
            }();
            return range;
        }

        bool iequals(std::string_view lhs, std::string_view rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i])))
                    return false;
            }
            return true;
        }

    }

    class IopEmulator::Impl final : public IopGuestExecutor
    {
    public:
        using CpuState = IopCpuState;

        struct Module
        {
            int id = 0;
            std::string path;
            std::string name;
            uint32_t base = 0;
            uint32_t size = 0;
            uint32_t entry = 0;
            uint32_t gp = 0;
            bool resident = false;
        };

        struct GuestCallback
        {
            uint32_t function = 0;
            uint32_t gp = 0;
        };

        struct ScheduledGuestCallback
        {
            uint32_t function = 0u;
            uint32_t gp = 0u;
            uint32_t argument = 0u;
        };

        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              sysmem(host, memory),
              kernel(memory),
              cdvd(host, memory, kernel),
              vblank(kernel),
              rpc(host, memory, kernel),
              sysclib(memory),
              stdio(host, memory),
              heaplib(memory),
              intrman(memory),
              timrman(),
              ioman(memory),
              cpuCore(memory),
              imports(memory),
              loadcore(memory, imports)
        {
            reset();
        }

        void reset()
        {
            memory.reset();
            kernel.reset();
            modules.clear();
            imports.reset();
            rpc.reset();
            cdvd.reset();
            intrman.reset();
            timrman.reset();
            ioman.reset();
            pendingDmaInterrupts.clear();
            pendingGuestCallbacks.clear();
            nextModuleId = 1;
            moduleCursor = kModuleLoadBase;
            totalCycles = 0;
            totalInstructions = 0;
            eeCycleCarry = 0;
            activeCpu = nullptr;
            lastError.clear();
            servicingDmaInterrupts = false;
            servicingGuestCallbacks = false;
            servicingSpuInterrupt = false;
            syncYieldDepth = 0u;
            callDepth = 0u;
            secrMcCommandHandler = {};
            secrMcDevIdHandler = {};
            checkKelfPathCallback = {};
        }

        uint8_t read8(uint32_t address) const
        {
            return memory.read8(address);
        }

        uint16_t read16(uint32_t address) const
        {
            return memory.read16(address);
        }

        uint32_t read32(uint32_t address) const
        {
            return memory.read32(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            memory.write8(address, value);
            schedulePendingDma();
        }

        void write16(uint32_t address, uint16_t value)
        {
            memory.write16(address, value);
            schedulePendingDma();
        }

        void write32(uint32_t address, uint32_t value)
        {
            memory.write32(address, value);
            schedulePendingDma();
        }

        void schedulePendingDma()
        {
            if (const auto dma = memory.takeDmaStart())
                pendingDmaInterrupts[dma->irq] = totalCycles + dma->delayCycles;
        }

        bool readRam(uint32_t address, void *destination, size_t size) const
        {
            return memory.readRam(address, destination, size);
        }

        bool writeRam(uint32_t address, const void *source, size_t size)
        {
            return memory.writeRam(address, source, size);
        }

        bool zeroRam(uint32_t address, size_t size)
        {
            return memory.zeroRam(address, size);
        }

        bool isHardwareAddress(uint32_t phys) const
        {
            return memory.isHardwareAddress(phys);
        }

        uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt)
        {
            return memory.allocate(size, alignment, fixed);
        }

        bool freeAllocation(uint32_t address)
        {
            return memory.freeAllocation(address);
        }

        void log(LogLevel level, std::string_view text)
        {
            host.log(level, text);
        }

        bool checkInterrupt(CpuState &cpu)
        {
            const uint32_t status = cpu.cop0[12];
            if ((status & 1u) == 0u)
                return false;
            if ((status & 0x2u) != 0u)
                return false;
            const bool pending = memory.interruptControl() != 0u && (memory.interruptStatus() & memory.interruptMask()) != 0u;
            if (!pending)
                return false;
            cpu.cop0[13] |= 0x400u;
            cpuCore.raiseException(cpu, 0u, cpu.pc, false);
            return true;
        }

        enum class ImportDisposition
        {
            Handled,
            JumpToGuest,
            Missing,
        };

        ImportDisposition dispatchImport(const IopImportCall &call, CpuState &cpu)
        {
            const uint32_t a0 = cpu.gpr[4];
            auto setV0 = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };

            if (traceImport(call.library))
            {
                std::ostringstream out;
                out << "[IOP import] " << call.library << ':' << std::dec << call.ordinal << std::hex
                    << " a0=0x" << cpu.gpr[4] << " a1=0x" << cpu.gpr[5] << " a2=0x" << cpu.gpr[6]
                    << " a3=0x" << cpu.gpr[7] << " ra=0x" << cpu.gpr[31] << std::dec
                    << " th=" << (kernel.currentThread() ? kernel.currentThread()->id : -1);
                log(LogLevel::Info, out.str());
            }

            if (kernel.currentThread() == nullptr && dispatchBlockingOutsideThread(call, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "sysmem") && sysmem.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "cdvdman") && cdvd.dispatchImport(call.ordinal, cpu))
            {
                if (const auto callback = cdvd.takeCompletionCallback())
                {
                    const uint64_t delay = call.ordinal == 6u // sceCdRead
                                               ? std::max<uint64_t>(kCdvdCompletionCycles,
                                                                    static_cast<uint64_t>(cpu.gpr[5]) * kCdvdCyclesPerSector)
                                               : kCdvdCompletionCycles;
                    pendingGuestCallbacks.emplace(
                        totalCycles + delay,
                        ScheduledGuestCallback{
                            callback->address,
                            callback->gp,
                            callback->reason,
                        });
                }
                return ImportDisposition::Handled;
            }

            if (iequals(call.library, "loadcore") && loadcore.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "thbase") || iequals(call.library, "threadman"))
            {
                return kernel.dispatchThreadImport(call.ordinal, cpu, totalCycles)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thsemap"))
            {
                return kernel.dispatchSemaphoreImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thevent"))
            {
                return kernel.dispatchEventImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "sifcmd"))
            {
                return rpc.dispatchSifCmdImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "intrman") && intrman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "secrman"))
            {
                switch (call.ordinal)
                {
                case 4: // SecrSetMcCommandHandler
                    secrMcCommandHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                case 5: // SecrSetMcDevIDHandler
                    secrMcDevIdHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                default:
                    break;
                }
            }
            if (iequals(call.library, "modload") && call.ordinal == 13u)
            {
                checkKelfPathCallback = {a0, cpu.gpr[28]};
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "ioman") && ioman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sifman"))
            {
                return rpc.dispatchSifManImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "vblank") && vblank.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "timrman") && timrman.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "dmacman"))
            {
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "stdio") && stdio.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sysclib"))
            {
                return sysclib.dispatchImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "heaplib") && heaplib.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            const uint32_t target = imports.resolve(call.library, call.ordinal, call.version);
            if (target != 0u)
            {
                cpu.pc = target;
                cpu.branchPending = false;
                return ImportDisposition::JumpToGuest;
            }

            std::ostringstream out;
            out << "[IOP] unhandled import " << call.library << ':' << call.ordinal
                << " version=0x" << std::hex << call.version << " pc=0x" << cpu.pc;
            log(LogLevel::Warning, out.str());
            setV0(0);
            return ImportDisposition::Missing;
        }

        void tracePcStep(const CpuState &cpu)
        {
            static uint64_t traced = 0u;
            if (traced >= tracePc.limit)
                return;
            ++traced;
            char line[200];
            std::snprintf(line, sizeof(line),
                          "[IOP pc] %06x %08x v0=%x v1=%x a0=%x a1=%x a2=%x a3=%x s0=%x s1=%x sp=%x ra=%x",
                          cpu.pc, memory.read32(cpu.pc), cpu.gpr[2], cpu.gpr[3], cpu.gpr[4], cpu.gpr[5], cpu.gpr[6],
                          cpu.gpr[7], cpu.gpr[16], cpu.gpr[17], cpu.gpr[29], cpu.gpr[31]);
            log(LogLevel::Info, line);
        }

        bool step(CpuState &cpu)
        {
            if (cpu.stopped)
                return false;
            if (cpu.pc == kThreadReturnSentinel || cpu.pc == kCallReturnSentinel)
            {
                cpu.stopped = true;
                return false;
            }
            if (physicalAddress(cpu.pc) >= kRamSize)
            {
                std::ostringstream out;
                out << "[IOP] execution outside RAM pc=0x" << std::hex << cpu.pc;
                log(LogLevel::Error, out.str());
                cpu.stopped = true;
                return false;
            }
            if (checkInterrupt(cpu))
                return true;

            if (const auto import = imports.decode(cpu.pc))
            {
                const ImportDisposition disposition = dispatchImport(*import, cpu);
                ++totalInstructions;
                ++totalCycles;
                if (disposition == ImportDisposition::JumpToGuest)
                    return true;
                cpu.pc = cpu.gpr[31];
                cpu.branchPending = false;
                return !cpu.stopped;
            }

            if (tracePc.active && cpu.pc >= tracePc.low && cpu.pc < tracePc.high)
                tracePcStep(cpu);
            const bool running = cpuCore.executeInstruction(cpu);
            schedulePendingDma();
            ++totalInstructions;
            ++totalCycles;
            return running;
        }

        uint32_t runCpu(CpuState &cpu, uint32_t instructionBudget)
        {
            CpuState *previous = activeCpu;
            activeCpu = &cpu;
            const uint64_t start = totalInstructions;
            while (!cpu.stopped && !cpu.yielded && totalInstructions - start < instructionBudget)
            {
                if (!step(cpu))
                    break;
                if (!servicingDmaInterrupts && !pendingDmaInterrupts.empty())
                    servicePendingDmaInterrupts();
                if (!servicingGuestCallbacks && !pendingGuestCallbacks.empty())
                    servicePendingGuestCallbacks();
                if (!servicingSpuInterrupt && memory.spu2().interruptPending())
                    serviceSpuInterrupt();
            }
            activeCpu = previous;
            return static_cast<uint32_t>(totalInstructions - start);
        }

        uint32_t callFunction(uint32_t address,
                              uint32_t a0,
                              uint32_t a1,
                              uint32_t a2,
                              uint32_t a3,
                              uint32_t gp,
                              uint32_t budget = kMaxCallInstructions)
        {
            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };

            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            if (depth < kCallStackCapacity)
            {
                const uint32_t stackTop = kCallStackLimit - depth * kCallStackSize;
                cpu.gpr[29] = stackTop - 32u;
            }
            else if (activeCpu && activeCpu->gpr[29] > kCallStackBase + kStackGuardBytes)
            {
                // Extremely deep re-entrancy borrows unused space below the
                // suspended caller's live frame. Stack growth remains away
                // from the caller, so its saved registers stay intact.
                cpu.gpr[29] = (activeCpu->gpr[29] - kStackGuardBytes) & ~15u;
            }
            else
            {
                cpu.gpr[29] = kCallStackBase - 32u;
            }
            cpu.gpr[31] = kCallReturnSentinel;
            runCpu(cpu, budget);
            return cpu.gpr[2];
        }

        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t gp) override
        {
            return callFunction(address, a0, a1, a2, a3, gp);
        }

        uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t a2,
                                                uint32_t a3,
                                                uint32_t gp,
                                                uint32_t instructionBudget) override
        {
            return callFunction(address, a0, a1, a2, a3, gp, instructionBudget);
        }

        void runThreadsAbove(uint32_t priority) override
        {
            // Bounded: a higher-priority busy-wait must not stall the caller.
            constexpr uint64_t kMaxCycles = 0x20000u;
            if (kernel.currentThread() != nullptr)
                return;
            const uint64_t limit = totalCycles + kMaxCycles;
            while (totalCycles < limit && kernel.hasReadyAbove(priority))
                scheduleUntil(std::min(limit, totalCycles + kDefaultSlice));
        }

        // Not that good to use exception handling for control flow but will do for now
        void servicePendingDmaInterrupts()
        {
            if (servicingDmaInterrupts || pendingDmaInterrupts.empty())
                return;

            servicingDmaInterrupts = true;

            std::vector<int> completed;
            for (auto it = pendingDmaInterrupts.begin(); it != pendingDmaInterrupts.end();)
            {
                if (it->second > totalCycles)
                {
                    ++it;
                    continue;
                }
                completed.push_back(it->first);
                it = pendingDmaInterrupts.erase(it);
            }
            try
            {
                for (const int irq : completed)
                {
                    if (irq == kDmaSpu0CompletionIrq || irq == kDmaSpu1CompletionIrq)
                        memory.spu2().dmaComplete(irq == kDmaSpu1CompletionIrq ? 1 : 0);
                    (void)intrman.dispatchInterrupt(irq, *this);
                }
            }
            catch (...)
            {
                servicingDmaInterrupts = false;
                throw;
            }
            servicingDmaInterrupts = false;
        }

        void serviceSpuInterrupt()
        {
            if (servicingSpuInterrupt || !memory.spu2().interruptPending())
                return;
            const uint32_t admaDone = memory.spu2().takeAdmaCompletions();
            const bool spuIrq = memory.spu2().takeInterrupt();
            servicingSpuInterrupt = true;
            try
            {
                for (int core = 0; core < 2; ++core)
                {
                    if ((admaDone & (1u << core)) == 0u)
                        continue;
                    memory.spu2().dmaComplete(core);
                    (void)intrman.dispatchInterrupt(core ? kDmaSpu1CompletionIrq : kDmaSpu0CompletionIrq, *this);
                }
                if (spuIrq)
                    (void)intrman.dispatchInterrupt(IopSpu2::SpuInterruptLine, *this);
            }
            catch (...)
            {
                servicingSpuInterrupt = false;
                throw;
            }
            servicingSpuInterrupt = false;
        }

        void servicePendingGuestCallbacks()
        {
            if (servicingGuestCallbacks || pendingGuestCallbacks.empty())
                return;

            std::vector<ScheduledGuestCallback> callbacks;
            for (auto it = pendingGuestCallbacks.begin(); it != pendingGuestCallbacks.end();)
            {
                if (it->first > totalCycles)
                    break;
                callbacks.push_back(it->second);
                it = pendingGuestCallbacks.erase(it);
            }
            if (callbacks.empty())
                return;

            servicingGuestCallbacks = true;
            try
            {
                for (const ScheduledGuestCallback &callback : callbacks)
                {
                    if (callback.function != 0u)
                    {
                        (void)callFunction(callback.function,
                                           callback.argument,
                                           0u,
                                           0u,
                                           0u,
                                           callback.gp,
                                           100000u);
                    }
                }
            }
            catch (...)
            {
                servicingGuestCallbacks = false;
                throw;
            }
            servicingGuestCallbacks = false;
        }

        // ATV_IOP_THREAD_DUMP=1 prints thread states once per IOP second;
        // ATV_IOP_DUMP_ADDR=addr,size adds a RAM word dump.
        void debugDumpThreads()
        {
            static const bool enabled = []
            {
                const char *value = std::getenv("ATV_IOP_THREAD_DUMP");
                return value && value[0] == '1';
            }();
            if (!enabled || totalCycles < nextThreadDumpCycle)
                return;
            nextThreadDumpCycle = totalCycles + kIopClockHz;
            std::ostringstream out;
            out << "[IOP threads] cycle=" << totalCycles;
            for (const auto &[id, thread] : kernel.threads())
            {
                out << " | " << id << ":s" << static_cast<int>(thread.state) << " pc=0x" << std::hex << thread.cpu.pc
                    << " ra=0x" << thread.cpu.gpr[31] << " w=" << std::dec << thread.waitId << " p=" << thread.priority;
            }
            if (const char *dump = std::getenv("ATV_IOP_DUMP_ADDR"))
            {
                uint32_t address = static_cast<uint32_t>(std::strtoul(dump, nullptr, 16));
                const char *comma = std::strchr(dump, ',');
                const uint32_t size = comma ? static_cast<uint32_t>(std::strtoul(comma + 1, nullptr, 16)) : 0x40u;
                out << " | mem@0x" << std::hex << address << ':';
                for (uint32_t offset = 0; offset < size; offset += 4u)
                    out << ' ' << memory.read32(address + offset);
            }
            log(LogLevel::Info, out.str());
        }

        void runCycles(uint64_t cycles) noexcept
        {
            try
            {
                debugDumpThreads();
                memory.spu2().advanceIopCycles(cycles);
                scheduleUntil(totalCycles + cycles);
            }
            catch (...)
            {
                // Runtime scheduling must never throw through EeScheduler::accountCycles().
            }
        }

        // Guest code run outside any thread (RPC server functions) cannot block.
        // A driver that sleeps or waits there for one of its own threads (989SND
        // posting a CD read to its reader thread) would spin forever, so let the
        // other threads run for a while instead of returning immediately.
        void yieldFromGuestCall(uint64_t cycles)
        {
            if (syncYieldDepth >= 2u)
                return;
            ++syncYieldDepth;
            try
            {
                scheduleUntil(totalCycles + cycles);
            }
            catch (...)
            {
                --syncYieldDepth;
                throw;
            }
            --syncYieldDepth;
        }

        bool dispatchBlockingOutsideThread(const IopImportCall &call, CpuState &cpu)
        {
            constexpr uint64_t kYieldCycles = 4096u;
            constexpr int kMaxSemaphoreYields = 256;
            if (iequals(call.library, "thbase") && call.ordinal == 24u) // SleepThread
            {
                yieldFromGuestCall(kYieldCycles);
                cpu.gpr[2] = 0u;
                return true;
            }
            if (iequals(call.library, "thbase") && call.ordinal == 33u) // DelayThread
            {
                const uint64_t delay = (static_cast<uint64_t>(cpu.gpr[4]) * kIopClockHz) / 1'000'000ull;
                yieldFromGuestCall(std::clamp<uint64_t>(delay, 1u, kIopClockHz / 60u));
                cpu.gpr[2] = 0u;
                return true;
            }
            if (iequals(call.library, "thsemap") && call.ordinal == 8u) // WaitSema
            {
                for (int attempt = 0; attempt < kMaxSemaphoreYields; ++attempt)
                {
                    CpuState poll{};
                    poll.gpr[4] = cpu.gpr[4];
                    if (kernel.dispatchSemaphoreImport(9u, poll) && poll.gpr[2] != static_cast<uint32_t>(-419))
                    {
                        cpu.gpr[2] = poll.gpr[2];
                        return true;
                    }
                    yieldFromGuestCall(kYieldCycles);
                }
                cpu.gpr[2] = static_cast<uint32_t>(-419);
                return true;
            }
            return false;
        }

        void scheduleUntil(uint64_t target)
        {
            {
                while (totalCycles < target)
                {
                    serviceSpuInterrupt();
                    servicePendingDmaInterrupts();
                    servicePendingGuestCallbacks();
                    if (const uint32_t timerIrqs = timrman.serviceDue(totalCycles, *this))
                    {
                        for (int irq = 0; irq < 32; ++irq)
                        {
                            if ((timerIrqs & (1u << irq)) != 0u)
                                (void)intrman.dispatchInterrupt(irq, *this);
                        }
                    }
                    uint64_t nextWake = target;
                    IopThread *next = kernel.beginNextReady(totalCycles, target, nextWake);
                    if (!next)
                    {
                        for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                            nextWake = std::min(nextWake, completionCycle);
                        if (!pendingGuestCallbacks.empty())
                            nextWake = std::min(nextWake, pendingGuestCallbacks.begin()->first);
                        nextWake = timrman.nextEventCycle(nextWake);
                        totalCycles = std::max(totalCycles + 1u, std::min(target, nextWake));
                        continue;
                    }
                    const uint64_t before = totalCycles;
                    runCpu(next->cpu, static_cast<uint32_t>(std::min<uint64_t>(kDefaultSlice, target - totalCycles)));
                    kernel.endTimeslice(*next, kThreadReturnSentinel);
                    if (totalCycles == before)
                        ++totalCycles;
                }
            }
        }

        ModuleLoadResult loadImage(std::string path, std::span<const uint8_t> image, const void *arguments, uint32_t argumentSize)
        {
            ModuleLoadResult result{true, -1, -1};
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, moduleCursor);
            moduleCursor = loaded.nextModuleCursor;
            if (!loaded)
            {
                if (loaded.error == IopImageLoadError::InvalidElf)
                    log(LogLevel::Error, "[IOP] rejected invalid/non-MIPS IRX ELF");
                else if (loaded.error == IopImageLoadError::ArenaExhausted)
                    log(LogLevel::Error, "[IOP] module arena exhausted");
                return result;
            }
            if (!loaded.relocationsComplete)
                log(LogLevel::Warning, "[IOP] one or more IRX relocations were unsupported");

            Module module;
            module.id = nextModuleId++;
            module.path = std::move(path);
            const size_t slash = module.path.find_last_of("/\\:");
            module.name = slash == std::string::npos ? module.path : module.path.substr(slash + 1u);
            module.base = loaded.base;
            module.size = loaded.size;
            module.entry = loaded.entry;
            module.gp = loaded.gp;

            uint32_t args = 0u;
            if (arguments && argumentSize)
            {
                args = allocate(argumentSize + 1u, 16u);
                if (args)
                {
                    writeRam(args, arguments, argumentSize);
                    write8(args + argumentSize, 0u);
                }
            }
            const uint32_t startResult = callFunction(module.entry, argumentSize, args, 0u, 0u, module.gp);
            if (args)
                freeAllocation(args);
            module.resident = startResult == 0u || startResult == 2u;
            result.moduleId = module.id;
            result.startResult = static_cast<int32_t>(startResult);
            modules[module.id] = std::move(module);

            std::ostringstream out;
            out << "[IOP] loaded IRX id=" << result.moduleId
                << " entry=0x" << std::hex << modules[result.moduleId].entry
                << " base=0x" << modules[result.moduleId].base
                << " start=" << std::dec << result.startResult;
            log(LogLevel::Info, out.str());
            return result;
        }

        ModuleLoadResult loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readWholeHostFile(host, path, image))
            {
                log(LogLevel::Warning, std::string("[IOP] failed to open IRX '") + std::string(path) + "'");
                return {true, -1, -1};
            }
            return loadImage(std::string(path), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromGuest(host, guestAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "buffer@0x" << std::hex << guestAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        bool stopModule(int32_t moduleId, int32_t *result)
        {
            auto it = modules.find(moduleId);
            if (it == modules.end())
                return false;
            // A removable IRX normally exposes a stop entry through module metadata. We do not guess it; terminate owned execution and release the image cleanly.
            kernel.terminateThreadsInRange(it->second.base, it->second.size);
            rpc.removeServersInRange(it->second.base, it->second.size);
            imports.eraseRange(it->second.base, it->second.size);
            modules.erase(it);
            kernel.cleanupDeadThreads();
            if (result)
                *result = 0;
            return true;
        }

        IopHost &host;
        IopMemory memory;
        IopSysmem sysmem;
        IopKernel kernel;
        IopCdvd cdvd;
        IopVblank vblank;
        IopRpcBridge rpc;
        IopSysclib sysclib;
        IopStdio stdio;
        IopHeaplib heaplib;
        IopIntrman intrman;
        IopTimrman timrman;
        IopIoman ioman;
        IopCpuCore cpuCore;
        IopImportRegistry imports;
        IopLoadcore loadcore;
        std::map<int, Module> modules;
        std::map<int, uint64_t> pendingDmaInterrupts;
        std::multimap<uint64_t, ScheduledGuestCallback> pendingGuestCallbacks;
        uint32_t nextModuleId = 1;
        uint32_t moduleCursor = kModuleLoadBase;
        uint64_t totalCycles = 0;
        uint64_t totalInstructions = 0;
        uint64_t eeCycleCarry = 0;
        CpuState *activeCpu = nullptr;
        std::string lastError;
        bool servicingDmaInterrupts = false;
        bool servicingGuestCallbacks = false;
        bool servicingSpuInterrupt = false;
        uint32_t syncYieldDepth = 0u;
        uint64_t nextThreadDumpCycle = 0u;
        uint32_t callDepth = 0u;
        const TracePcRange tracePc = tracePcRange();
        GuestCallback secrMcCommandHandler;
        GuestCallback secrMcDevIdHandler;
        GuestCallback checkKelfPathCallback;
    };

    IopEmulator::IopEmulator(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
    }

    IopEmulator::~IopEmulator() = default;

    void IopEmulator::reset()
    {
        m_impl->reset();
    }

    ModuleLoadResult IopEmulator::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModule(path, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    bool IopEmulator::stopModule(int32_t moduleId, int32_t *result)
    {
        return m_impl->stopModule(moduleId, result);
    }

    void IopEmulator::runEeCycles(uint64_t eeCycles) noexcept
    {
        const uint64_t total = m_impl->eeCycleCarry + eeCycles;
        const uint64_t iopCycles = total / 8u;
        m_impl->eeCycleCarry = total % 8u;
        if (iopCycles)
            m_impl->runCycles(iopCycles);
    }

    void IopEmulator::runIopCycles(uint64_t iopCycles) noexcept
    {
        if (iopCycles)
            m_impl->runCycles(iopCycles);
    }

    RpcResult IopEmulator::handleRpc(const RpcRequest &request)
    {
        return m_impl->rpc.handleRpc(request, *m_impl);
    }

    RpcResult IopEmulator::handleRpcBuffered(const RpcRequest &request,
                                             std::span<const uint8_t> send,
                                             std::vector<uint8_t> &receive)
    {
        return m_impl->rpc.handleRpcBuffered(request, *m_impl, send, receive);
    }

    bool IopEmulator::hasRpcServer(uint32_t sid) const noexcept
    {
        return m_impl->rpc.hasServer(sid);
    }

    void IopEmulator::onSifTransfer(const SifTransfer &transfer)
    {
        m_impl->rpc.onSifTransfer(transfer);
    }

    uint32_t IopEmulator::allocateMemory(uint32_t size, uint32_t alignment)
    {
        return m_impl->memory.allocate(size, alignment);
    }

    bool IopEmulator::freeMemory(uint32_t address)
    {
        return m_impl->memory.freeAllocation(address);
    }

    bool IopEmulator::readMemory(uint32_t address, void *destination, size_t size) const
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.readRam(address, destination, size);
    }

    bool IopEmulator::writeMemory(uint32_t address, const void *source, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.writeRam(address, source, size);
    }

    bool IopEmulator::zeroMemory(uint32_t address, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.zeroRam(address, size);
    }

    bool IopEmulator::isMemoryRange(uint32_t address, size_t size) const
    {
        const bool physicalSegment = address < IopMemory::RamSize;
        const bool cachedSegment = address >= 0x80000000u && address < 0x80200000u;
        const bool uncachedSegment = address >= 0xA0000000u && address < 0xA0200000u;
        if (!physicalSegment && !cachedSegment && !uncachedSegment)
            return false;
        const uint32_t physical = IopMemory::physicalAddress(address);
        return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
    }

    uint32_t IopEmulator::renderAudio(int16_t *stereo, uint32_t maxFrames)
    {
        return m_impl->memory.spu2().renderPaced(stereo, maxFrames);
    }

    void IopEmulator::setAudioHostDriven(bool hostDriven) noexcept
    {
        m_impl->memory.spu2().setHostDriven(hostDriven);
    }

    uint64_t IopEmulator::cycles() const noexcept
    {
        return m_impl->totalCycles;
    }

    uint64_t IopEmulator::instructions() const noexcept
    {
        return m_impl->totalInstructions;
    }

    uint32_t IopEmulator::loadedModuleCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->modules.size());
    }

    uint32_t IopEmulator::threadCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->kernel.threadCount());
    }

    uint32_t IopEmulator::rpcServerCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->rpc.serverCount());
    }

}
