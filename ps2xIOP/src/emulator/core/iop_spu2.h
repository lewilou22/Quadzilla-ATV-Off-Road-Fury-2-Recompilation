#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ps2x::iop::detail
{
    // SPU2: two cores of 24 ADPCM voices over 2 MB of sound RAM. The IOP side
    // (register access, DMA ch4/ch7) runs on the IOP thread; samples are pulled
    // by the host audio thread through render(). Without a host consumer the
    // IOP clock drives the voices instead (advanceIopCycles), so drivers that
    // poll ENDX/ENVX still see voices finish.
    class IopSpu2
    {
    public:
        static constexpr uint32_t RegisterBase = 0x1F900000u;
        static constexpr uint32_t RegisterWindow = 0x800u;
        static constexpr uint32_t RamHalfwords = 0x100000u;
        static constexpr uint32_t SampleRate = 48000u;
        static constexpr uint32_t IopCyclesPerSample = 768u;
        static constexpr int SpuInterruptLine = 9;

        IopSpu2();

        void reset();

        [[nodiscard]] static bool isRegister(uint32_t physical) noexcept
        {
            return physical >= RegisterBase && physical < RegisterBase + RegisterWindow;
        }

        [[nodiscard]] uint16_t read16(uint32_t physical);
        void write16(uint32_t physical, uint16_t value);

        // DMA channel 4 (core 0) and 7 (core 1). Data moves immediately; the
        // status bits flip back when the DMA interrupt is delivered. Returns
        // false for an auto-DMA (PCM input) transfer, which completes through
        // takeAdmaCompletions() once the mixer has played it.
        [[nodiscard]] bool dmaWrite(int core, const uint8_t *source, uint32_t bytes);
        void dmaRead(int core, uint8_t *destination, uint32_t bytes);
        void dmaComplete(int core);

        // SPU2 interrupt pending for IOP interrupt line 9.
        [[nodiscard]] bool takeInterrupt() noexcept
        {
            return m_irqPending.exchange(false, std::memory_order_acq_rel);
        }
        [[nodiscard]] bool interruptPending() const noexcept
        {
            return m_irqPending.load(std::memory_order_relaxed) ||
                   m_admaDone.load(std::memory_order_relaxed) != 0u;
        }
        // Bit n set: the auto-DMA transfer of core n has been played out.
        [[nodiscard]] uint32_t takeAdmaCompletions() noexcept
        {
            return m_admaDone.exchange(0u, std::memory_order_acq_rel);
        }

        // Interleaved stereo 16-bit at 48 kHz, regardless of the IOP clock.
        void render(int16_t *stereo, uint32_t frames);
        // Host audio thread: renders at most what IOP time has made due
        // (advanceIopCycles in host-driven mode) and returns the frame count.
        [[nodiscard]] uint32_t renderPaced(int16_t *stereo, uint32_t maxFrames);
        void setHostDriven(bool hostDriven) noexcept { m_hostDriven.store(hostDriven, std::memory_order_release); }
        void advanceIopCycles(uint64_t cycles);

    private:
        struct VolumeSlide
        {
            uint16_t reg = 0;
            int32_t value = 0;
            uint32_t counter = 0;

            void set(uint16_t raw);
            void update();
        };

        enum class EnvPhase : uint8_t
        {
            Stopped,
            Attack,
            Decay,
            Sustain,
            Release,
        };

        struct Voice
        {
            VolumeSlide volL;
            VolumeSlide volR;
            uint16_t pitch = 0;
            uint16_t adsr1 = 0;
            uint16_t adsr2 = 0;
            uint32_t ssa = 0;
            uint32_t lsa = 0;
            uint32_t block = 0;
            bool lsaWritten = false;
            EnvPhase phase = EnvPhase::Stopped;
            int32_t envelope = 0;
            uint32_t envCounter = 0;
            uint32_t counter = 0;
            int32_t prev1 = 0;
            int32_t prev2 = 0;
            uint32_t samplePos = 0;
            std::array<int16_t, 28> decoded{};
            std::array<int16_t, 4> history{};
            int32_t out = 0;
        };

        struct Core
        {
            std::array<Voice, 24> voices{};
            uint32_t pmon = 0;
            uint32_t non = 0;
            uint32_t vmixL = 0;
            uint32_t vmixR = 0;
            uint32_t vmixEL = 0;
            uint32_t vmixER = 0;
            uint16_t mmix = 0;
            uint16_t attr = 0;
            uint32_t irqa = 0;
            uint32_t tsa = 0;
            uint32_t endx = 0;
            uint16_t statx = 0;
            uint16_t admas = 0;
            uint32_t esa = 0;
            uint32_t eea = 0;
            VolumeSlide mvolL;
            VolumeSlide mvolR;
            int32_t evolL = 0;
            int32_t evolR = 0;
            int32_t avolL = 0;
            int32_t avolR = 0;
            int32_t bvolL = 0;
            int32_t bvolR = 0;
            uint32_t noiseCounter = 0;
            int32_t noiseLevel = 0;
            // Reverb runs at 24 kHz, alternating left/right each 48 kHz tick.
            std::array<std::array<int16_t, 128>, 2> reverbDown{};
            std::array<std::array<int16_t, 128>, 2> reverbUp{};
            uint32_t reverbPos = 0;
            // Auto-DMA input: blocks of 256 left then 256 right samples.
            std::vector<int16_t> admaData;
            size_t admaBlock = 0;
            uint32_t admaSample = 0;
        };

        [[nodiscard]] uint16_t readLocked(uint32_t offset);
        void writeLocked(uint32_t offset, uint16_t value);
        void writeCore(Core &core, int index, uint32_t reg, uint16_t value);
        void writeCoreExtra(Core &core, uint32_t reg, uint16_t value);
        void keyOn(Core &core, int coreIndex, uint32_t mask);
        void keyOff(Core &core, uint32_t mask);
        void setAttr(Core &core, int index, uint16_t value);
        void startBlock(Voice &voice);
        int16_t nextSample(Core &core, int coreIndex, int voiceIndex);
        void checkIrq(uint32_t address);
        void checkIrqRange(uint32_t start, uint32_t halfwords);
        void raiseIrq(int core);
        void stepEnvelope(Voice &voice);
        int32_t mixVoice(Core &core, int coreIndex, int voiceIndex, int32_t previousOut, int32_t &left, int32_t &right);
        void mixSample(int32_t &outLeft, int32_t &outRight);
        void readAdmaInput(Core &core, int index, int32_t &left, int32_t &right);
        [[nodiscard]] uint32_t reverbAddress(const Core &core, int index, uint32_t reg) const;
        void reverb(Core &core, int index, int32_t inLeft, int32_t inRight, int32_t &outLeft, int32_t &outRight);
        void renderLocked(int16_t *stereo, uint32_t frames);

        std::mutex m_mutex;
        std::vector<uint16_t> m_ram;
        std::array<uint16_t, RegisterWindow / 2u> m_regs{};
        std::array<Core, 2> m_cores{};
        uint16_t m_irqInfo = 0;
        uint32_t m_sampleClock = 0;
        uint64_t m_iopCycleCarry = 0;
        std::atomic<bool> m_irqPending{false};
        std::atomic<bool> m_hostDriven{false};
        // IOP-time sample count (IOP thread) and samples mixed (host thread).
        static constexpr uint64_t kMaxPacedLag = SampleRate / 10u;
        std::atomic<uint64_t> m_iopSampleClock{0u};
        uint64_t m_renderedSamples = 0u;
        std::atomic<uint32_t> m_admaDone{0u};
    };
}
