#include "../emulated_sound_service.h"
#include "../emulator/iop_emulator.h"
#include "../emulator/iop_emulator_const.h"
#include "ps2x/iop/iop_audio_output.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <utility>

namespace ps2x::iop
{
    namespace
    {
        // The SPU2 of the running emulated sound driver, read by the host audio thread.
        std::mutex g_audioMutex;
        detail::IopEmulator *g_audioSource = nullptr;
        std::atomic<bool> g_audioConsumer{false};
    }

    uint32_t renderEmulatedAudio(int16_t *stereo, uint32_t maxFrames)
    {
        std::lock_guard<std::mutex> lock(g_audioMutex);
        return g_audioSource ? g_audioSource->renderAudio(stereo, maxFrames) : 0u;
    }

    void setEmulatedAudioConsumer(bool active)
    {
        g_audioConsumer.store(active, std::memory_order_release);
        std::lock_guard<std::mutex> lock(g_audioMutex);
        if (g_audioSource)
            g_audioSource->setAudioHostDriven(active);
    }
}

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kSifRpcModeNowait = 0x1u;
        // The IOP runs in slices of 1 ms IOP time, paced by the host clock.
        constexpr uint64_t kSliceCycles = kIopClockHz / 1000u;
        // A host stall longer than this is dropped instead of caught up.
        constexpr uint64_t kMaxLagCycles = kIopClockHz / 5u;
        // A call to a server the driver has not registered yet runs the IOP
        // this long (IOP time) for it to appear.
        constexpr uint64_t kServerWaitCycles = kIopClockHz;
        constexpr auto kSyncCallTimeout = std::chrono::seconds(5);

        bool envFlag(const char *name, bool fallback)
        {
            const char *value = std::getenv(name);
            if (!value || value[0] == '\0')
                return fallback;
            return value[0] != '0';
        }

        class EmulatedSoundDriverService final : public IopService
        {
        public:
            EmulatedSoundDriverService(IopHost &host, EmulatedSoundDriverBindings bindings)
                : m_host(host), m_bindings(std::move(bindings))
            {
            }

            ~EmulatedSoundDriverService() override
            {
                shutdown();
            }

            [[nodiscard]] std::string_view name() const override
            {
                return m_bindings.serviceName;
            }

            [[nodiscard]] std::span<const uint32_t> sids() const override
            {
                return m_bindings.sids;
            }

            // An IOP reset unloads the driver; the next call boots it again.
            void reset() override
            {
                shutdown();
            }

            [[nodiscard]] RpcResult handleRpc(const RpcRequest &request) override
            {
                if (!ensureRunning())
                    return {};

                auto job = std::make_shared<Job>();
                job->request = request;
                if (request.send.address != 0u && request.send.size != 0u)
                {
                    job->send.resize(request.send.size);
                    if (!m_host.readGuest(request.send.address, job->send.data(), job->send.size()))
                        job->send.clear();
                }

                const bool nowait = (request.mode & kSifRpcModeNowait) != 0u;
                std::unique_lock<std::mutex> lock(m_mutex);
                writeFinishedNowaitRepliesLocked();
                m_jobs.push_back(job);
                if (nowait)
                    m_nowaitJobs.push_back(job);
                m_wake.notify_one();

                RpcResult result{};
                result.handled = true;
                result.resultAddress = request.receive.address;
                result.serverDispatchPolicy = ServerDispatchPolicy::Suppress;
                result.signalCompletion = true;
                if (nowait)
                    return result; // the reply reaches the EE on a later call, in order

                ++m_syncCalls;
                const auto waitStart = std::chrono::steady_clock::now();
                if (!m_done.wait_for(lock, kSyncCallTimeout, [&]
                                     { return job->done || m_stop; }) ||
                    !job->done)
                {
                    std::ostringstream out;
                    out << "[IOP:sound] " << m_bindings.serviceName << " call sid=0x" << std::hex << request.sid
                        << " rpc=0x" << request.function << " timed out";
                    m_host.log(LogLevel::Error, out.str());
                    return {};
                }
                m_syncWaitNs += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count());
                writeFinishedNowaitRepliesLocked();
                if (!job->result.handled)
                    return job->result;
                writeReply(*job);
                return result;
            }

        private:
            struct Job
            {
                RpcRequest request;
                std::vector<uint8_t> send;
                std::vector<uint8_t> receive;
                RpcResult result{};
                bool done = false;
            };

            // EE thread. Boots the emulated IOP with the driver modules and
            // starts its thread on first use.
            bool ensureRunning()
            {
                std::lock_guard<std::mutex> boot(m_bootMutex);
                if (m_thread.joinable())
                    return true;
                if (m_bootFailed)
                    return false;

                auto emulator = std::make_unique<IopEmulator>(m_host);
                for (const std::string &path : m_bindings.modules)
                {
                    ModuleLoadResult loaded{};
                    try
                    {
                        loaded = emulator->loadModule(path, nullptr, 0u);
                    }
                    catch (...)
                    {
                        loaded.moduleId = -1;
                    }
                    std::ostringstream out;
                    out << "[IOP:sound] " << m_bindings.serviceName << " load '" << path << "' id=" << loaded.moduleId
                        << " start=" << loaded.startResult;
                    m_host.log(loaded.moduleId > 0 ? LogLevel::Info : LogLevel::Error, out.str());
                    if (loaded.moduleId <= 0)
                    {
                        m_bootFailed = true;
                        return false;
                    }
                }

                m_emulator = std::move(emulator);
                {
                    std::lock_guard<std::mutex> audio(g_audioMutex);
                    m_emulator->setAudioHostDriven(g_audioConsumer.load(std::memory_order_acquire));
                    g_audioSource = m_emulator.get();
                }
                m_stop = false;
                m_thread = std::thread([this]
                                       { threadMain(); });
                return true;
            }

            void shutdown()
            {
                std::lock_guard<std::mutex> boot(m_bootMutex);
                if (m_thread.joinable())
                {
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        m_stop = true;
                    }
                    m_wake.notify_all();
                    m_done.notify_all();
                    m_thread.join();
                }
                {
                    std::lock_guard<std::mutex> audio(g_audioMutex);
                    if (g_audioSource == m_emulator.get())
                        g_audioSource = nullptr;
                }
                m_emulator.reset();
                std::lock_guard<std::mutex> lock(m_mutex);
                m_jobs.clear();
                m_nowaitJobs.clear();
                m_bootFailed = false;
            }

            // IOP thread: runs queued calls in order between slices, and keeps
            // IOP time in step with the host clock.
            void threadMain()
            {
                using Clock = std::chrono::steady_clock;
                const auto start = Clock::now();
                const bool stats = envFlag("ATV2_IOP_SOUND_STATS", false);
                auto nextStats = start + std::chrono::seconds(10);
                uint64_t busyNs = 0u;
                uint64_t calls = 0u;

                const auto hostCycles = [&]
                {
                    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
                    return static_cast<uint64_t>(static_cast<double>(ns) * (static_cast<double>(kIopClockHz) / 1e9));
                };

                std::unique_lock<std::mutex> lock(m_mutex);
                while (!m_stop)
                {
                    if (!m_jobs.empty())
                    {
                        std::shared_ptr<Job> job = std::move(m_jobs.front());
                        m_jobs.pop_front();
                        lock.unlock();
                        const auto busyStart = Clock::now();
                        runJob(*job);
                        busyNs += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - busyStart).count());
                        ++calls;
                        lock.lock();
                        job->done = true;
                        m_done.notify_all();
                        continue;
                    }

                    uint64_t target = hostCycles() - m_droppedCycles;
                    if (target > m_pacedCycles + kMaxLagCycles)
                    {
                        m_droppedCycles += target - m_pacedCycles - kSliceCycles;
                        target = m_pacedCycles + kSliceCycles;
                    }
                    if (target > m_pacedCycles)
                    {
                        const uint64_t cycles = std::min(target - m_pacedCycles, kSliceCycles);
                        lock.unlock();
                        const auto busyStart = Clock::now();
                        m_emulator->runIopCycles(cycles);
                        busyNs += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - busyStart).count());
                        m_pacedCycles += cycles;
                        lock.lock();
                    }
                    else
                    {
                        m_wake.wait_for(lock, std::chrono::milliseconds(1), [&]
                                        { return m_stop || !m_jobs.empty(); });
                    }

                    if (stats && Clock::now() >= nextStats)
                    {
                        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
                        std::ostringstream out;
                        out << "[IOP:sound] t=" << static_cast<uint64_t>(seconds) << "s busy="
                            << static_cast<double>(busyNs) / 1e8 << "% (last 10 s) calls=" << calls
                            << " sync=" << m_syncCalls << " syncWaitUs="
                            << (m_syncCalls ? m_syncWaitNs / 1000u / m_syncCalls : 0u)
                            << " dropped=" << m_droppedCycles / (kIopClockHz / 1000u) << "ms iopCycles="
                            << m_emulator->cycles() << " threads=" << m_emulator->threadCount();
                        m_host.log(LogLevel::Info, out.str());
                        busyNs = 0u;
                        calls = 0u;
                        nextStats += std::chrono::seconds(10);
                    }
                }
            }

            // IOP thread.
            void runJob(Job &job)
            {
                try
                {
                    // The driver registers its servers from its own thread once
                    // it runs; a call can arrive before that.
                    uint64_t waited = 0u;
                    while (!m_emulator->hasRpcServer(job.request.sid) && waited < kServerWaitCycles)
                    {
                        m_emulator->runIopCycles(kSliceCycles);
                        m_pacedCycles += kSliceCycles;
                        waited += kSliceCycles;
                    }
                    job.result = m_emulator->handleRpcBuffered(job.request, job.send, job.receive);
                }
                catch (...)
                {
                    job.result = {};
                    job.receive.clear();
                }
                if (!job.result.handled)
                {
                    const uint64_t key = (static_cast<uint64_t>(job.request.sid) << 32u) | job.request.function;
                    if (m_loggedUnhandled.insert(key).second)
                    {
                        std::ostringstream out;
                        out << "[IOP:sound] " << m_bindings.serviceName << " no server answered sid=0x" << std::hex
                            << job.request.sid << " rpc=0x" << job.request.function;
                        m_host.log(LogLevel::Warning, out.str());
                    }
                }

                // ATV2_IOP_SOUND_TRACE=1 logs every call: function, sizes, the
                // first send words and the first reply word.
                static const bool trace = envFlag("ATV2_IOP_SOUND_TRACE", false);
                if (trace)
                {
                    const auto word = [](const std::vector<uint8_t> &bytes, size_t index)
                    {
                        uint32_t value = 0u;
                        if (bytes.size() >= (index + 1u) * 4u)
                            std::memcpy(&value, bytes.data() + index * 4u, sizeof(value));
                        return value;
                    };
                    std::ostringstream out;
                    out << "[IOP:sound] call sid=0x" << std::hex << job.request.sid << " rpc=0x" << job.request.function
                        << " mode=" << job.request.mode << std::dec << " send=" << job.send.size() << " [" << std::hex;
                    for (size_t i = 0; i < 6u && (i + 1u) * 4u <= job.send.size(); ++i)
                        out << (i ? " " : "") << word(job.send, i);
                    out << "] -> handled=" << job.result.handled << " reply=0x" << word(job.receive, 0) << std::dec
                        << " iopCycles=" << m_emulator->cycles();
                    m_host.log(LogLevel::Info, out.str());
                }
            }

            // EE thread, m_mutex held: nowait replies land in the EE in call order.
            void writeFinishedNowaitRepliesLocked()
            {
                while (!m_nowaitJobs.empty() && m_nowaitJobs.front()->done)
                {
                    if (m_nowaitJobs.front()->result.handled)
                        writeReply(*m_nowaitJobs.front());
                    m_nowaitJobs.pop_front();
                }
            }

            void writeReply(const Job &job)
            {
                const GuestBuffer &receive = job.request.receive;
                if (receive.address == 0u || job.receive.empty())
                    return;
                const size_t size = std::min<size_t>(receive.size, job.receive.size());
                (void)m_host.writeGuest(receive.address, job.receive.data(), size);
            }

            IopHost &m_host;
            EmulatedSoundDriverBindings m_bindings;
            std::mutex m_bootMutex;
            bool m_bootFailed = false;
            std::unique_ptr<IopEmulator> m_emulator;
            std::thread m_thread;

            std::mutex m_mutex;
            std::condition_variable m_wake;
            std::condition_variable m_done;
            std::deque<std::shared_ptr<Job>> m_jobs;
            std::deque<std::shared_ptr<Job>> m_nowaitJobs;
            bool m_stop = false;
            uint64_t m_syncCalls = 0u;
            uint64_t m_syncWaitNs = 0u;

            // IOP thread only.
            uint64_t m_pacedCycles = 0u;
            uint64_t m_droppedCycles = 0u;
            std::unordered_set<uint64_t> m_loggedUnhandled;
        };
    }

    std::unique_ptr<IopService> createEmulatedSoundDriverService(IopHost &host, EmulatedSoundDriverBindings bindings)
    {
        return std::make_unique<EmulatedSoundDriverService>(host, std::move(bindings));
    }
}
