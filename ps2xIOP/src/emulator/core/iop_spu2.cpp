#include "iop_spu2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kAddressMask = IopSpu2::RamHalfwords - 1u;
        constexpr uint8_t kFlagLoopEnd = 1u << 0u;
        constexpr uint8_t kFlagLoop = 1u << 1u;
        constexpr uint8_t kFlagLoopStart = 1u << 2u;
        constexpr uint16_t kAttrIrqEnable = 1u << 6u;
        constexpr uint16_t kAttrFxEnable = 1u << 7u;
        constexpr uint16_t kStatxDmaReady = 0x80u;
        constexpr uint16_t kStatxDmaBusy = 0x400u;
        // Render in short slices so IOP register writes never wait long for the lock.
        constexpr uint32_t kRenderSlice = 32u;

        constexpr int32_t kAdpcmFilters[5][2] = {
            {0, 0},
            {60, 0},
            {115, -52},
            {98, -55},
            {122, -60},
        };

        // Half-band filter of the reverb's 48 <-> 24 kHz resampling.
        constexpr uint32_t kReverbTaps = 39u;
        constexpr int16_t kReverbFilter[kReverbTaps] = {
            -1, 0, 2, 0, -10, 0, 35, 0, -103, 0, 266, 0, -616, 0, 1332, 0, -2960, 0, 10246, 16384,
            10246, 0, -2960, 0, 1332, 0, -616, 0, 266, 0, -103, 0, 35, 0, -10, 0, 2, 0, -1,
        };

        using GaussTable = std::array<std::array<int16_t, 4>, 256>;

        // The hardware's 4-tap interpolation kernel (nocash/Near formulation),
        // normalised so the four weights of every phase sum to 0x7F80.
        GaussTable buildGaussTable()
        {
            constexpr double kPi = 3.14159265358979323846;
            double table[512] = {};
            for (uint32_t n = 0; n < 512u; ++n)
            {
                const double k = 0.5 + n;
                const double s = std::sin(kPi * k * 2.048 / 1024.0);
                const double t = (std::cos(kPi * k * 2.000 / 1023.0) - 1.0) * 0.50;
                const double u = (std::cos(kPi * k * 4.000 / 1023.0) - 1.0) * 0.08;
                table[511u - n] = s * (t + u + 1.0) / k;
            }
            double sum = 0.0;
            for (const double value : table)
                sum += value;
            const double scale = 0x7F80 * 128 / sum;
            for (double &value : table)
                value *= scale;

            GaussTable result{};
            for (uint32_t phase = 0; phase < 256u; ++phase)
            {
                const double phaseSum = table[phase] + table[phase + 256u] + table[511u - phase] + table[255u - phase];
                const double diff = (phaseSum - 0x7F80) / 4.0;
                auto &weights = result[255u - phase];
                weights[0] = static_cast<int16_t>(std::lround(table[phase] - diff));
                weights[1] = static_cast<int16_t>(std::lround(table[phase + 256u] - diff));
                weights[2] = static_cast<int16_t>(std::lround(table[511u - phase] - diff));
                weights[3] = static_cast<int16_t>(std::lround(table[255u - phase] - diff));
            }
            return result;
        }

        const GaussTable &gaussTable()
        {
            static const GaussTable table = buildGaussTable();
            return table;
        }

        bool traceEnabled()
        {
            static const bool enabled = []
            {
                const char *value = std::getenv("ATV_SPU_TRACE");
                return value && value[0] != '\0' && value[0] != '0';
            }();
            return enabled;
        }

        int32_t clamp16(int32_t value)
        {
            return std::clamp<int32_t>(value, -0x8000, 0x7FFF);
        }

        int32_t applyVolume(int32_t sample, int32_t volume)
        {
            return (sample * volume) >> 15;
        }

        // Shared envelope stepper of ADSR and volume sweeps.
        struct EnvelopeStep
        {
            uint32_t counterIncrement;
            int32_t levelIncrement;
        };

        EnvelopeStep envelopeStep(bool decrease, bool exponential, int shift, int step, int32_t level)
        {
            uint32_t counterIncrement = 0x8000u >> std::max(0, shift - 11);
            int32_t levelIncrement = step * (1 << std::max(0, 11 - shift));
            if (exponential)
            {
                if (!decrease && level > 0x6000)
                    counterIncrement >>= 2u;
                if (decrease)
                    levelIncrement = static_cast<int16_t>((levelIncrement * level) >> 15);
            }
            return {counterIncrement, levelIncrement};
        }

        uint32_t setHigh(uint32_t value, uint16_t high)
        {
            return (static_cast<uint32_t>(high & 0xFu) << 16u) | (value & 0xFFFFu);
        }

        uint32_t setLow(uint32_t value, uint16_t low)
        {
            return (value & 0xF0000u) | low;
        }

        uint32_t setVoiceMaskLow(uint32_t value, uint16_t low)
        {
            return (value & 0xFF0000u) | low;
        }

        uint32_t setVoiceMaskHigh(uint32_t value, uint16_t high)
        {
            return (value & 0xFFFFu) | (static_cast<uint32_t>(high & 0xFFu) << 16u);
        }
    }

    void IopSpu2::VolumeSlide::set(uint16_t raw)
    {
        reg = raw;
        if ((raw & 0x8000u) == 0u)
            value = static_cast<int16_t>(static_cast<uint16_t>(raw << 1u));
    }

    void IopSpu2::VolumeSlide::update()
    {
        if ((reg & 0x8000u) == 0u)
            return;
        const bool exponential = (reg & 0x4000u) != 0u;
        const bool decrease = (reg & 0x2000u) != 0u;
        const bool negative = (reg & 0x1000u) != 0u;
        const int shift = (reg >> 2u) & 0x1Fu;
        const int rate = reg & 3u;
        int step = 7 - rate;
        if (decrease)
            step = ~step;

        EnvelopeStep delta = envelopeStep(decrease, exponential, shift, step, value);
        if (!(rate == 3 && shift == 0x1F))
            delta.counterIncrement = std::max<uint32_t>(1u, delta.counterIncrement);
        if (!(exponential && decrease) && negative)
            delta.levelIncrement = -delta.levelIncrement;

        counter += delta.counterIncrement;
        if (counter < 0x8000u)
            return;
        counter = 0;
        if (!decrease)
        {
            value = clamp16(value + delta.levelIncrement);
            return;
        }
        int32_t low = negative ? -0x8000 : 0;
        int32_t high = negative ? 0 : 0x7FFF;
        if (exponential)
        {
            low = 0;
            high = 0x7FFF;
        }
        value = std::clamp(value + delta.levelIncrement, low, high);
    }

    IopSpu2::IopSpu2()
        : m_ram(RamHalfwords)
    {
        (void)gaussTable();
        reset();
    }

    void IopSpu2::reset()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        m_regs.fill(0u);
        for (int index = 0; index < 2; ++index)
        {
            Core &core = m_cores[index];
            core = Core{};
            core.statx = kStatxDmaReady;
            core.endx = 0xFFFFFFu;
            core.mmix = index ? 0xFFCu : 0xFF0u;
            core.vmixL = core.vmixR = core.vmixEL = core.vmixER = 0xFFFFFFu;
            core.avolL = core.avolR = 0x7FFF;
            core.bvolL = core.bvolR = 0x7FFF;
            core.irqa = 0x800u;
            core.esa = index ? 0xFFFF8u : 0xEFFF8u;
            core.eea = index ? 0xFFFFFu : 0xEFFFFu;
            for (Voice &voice : core.voices)
            {
                voice.pitch = 0x3FFFu;
                voice.ssa = voice.lsa = voice.block = 0x2800u;
            }
        }
        m_irqInfo = 0u;
        m_sampleClock = 0u;
        m_iopCycleCarry = 0u;
        m_iopSampleClock.store(0u, std::memory_order_relaxed);
        m_renderedSamples = 0u;
        m_irqPending.store(false, std::memory_order_relaxed);
    }

    uint16_t IopSpu2::read16(uint32_t physical)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return readLocked(physical - RegisterBase);
    }

    void IopSpu2::write16(uint32_t physical, uint16_t value)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        writeLocked(physical - RegisterBase, value);
    }

    uint16_t IopSpu2::readLocked(uint32_t offset)
    {
        offset &= RegisterWindow - 2u;
        if (offset >= 0x760u && offset < 0x7B0u)
        {
            const Core &core = m_cores[(offset - 0x760u) / 0x28u];
            switch (0x760u + (offset - 0x760u) % 0x28u)
            {
            case 0x770u:
                return static_cast<uint16_t>(core.mvolL.value);
            case 0x772u:
                return static_cast<uint16_t>(core.mvolR.value);
            default:
                return m_regs[offset >> 1u];
            }
        }
        if (offset == 0x7C2u)
            return m_irqInfo;
        if (offset >= 0x7B0u)
            return m_regs[offset >> 1u];

        const int coreIndex = (offset >> 10u) & 1;
        Core &core = m_cores[coreIndex];
        const uint32_t reg = offset & 0x3FFu;
        if (reg < 0x180u)
        {
            const Voice &voice = core.voices[reg >> 4u];
            switch ((reg & 0xFu) >> 1u)
            {
            case 5:
                return static_cast<uint16_t>(voice.envelope);
            case 6:
                return static_cast<uint16_t>(voice.volL.value);
            case 7:
                return static_cast<uint16_t>(voice.volR.value);
            default:
                return m_regs[offset >> 1u];
            }
        }
        if (reg >= 0x1C0u && reg < 0x2E0u)
        {
            const Voice &voice = core.voices[(reg - 0x1C0u) / 12u];
            const uint32_t field = ((reg - 0x1C0u) % 12u) >> 1u;
            const uint32_t nax = (voice.block & 0xFFFF8u) + 1u + std::min<uint32_t>(voice.samplePos >> 2u, 6u);
            if (traceEnabled() && field == 5u)
                std::fprintf(stderr, "[spu2] r c%d v%02u nax=%05x\n", coreIndex, (reg - 0x1C0u) / 12u, nax);
            if (field == 4u)
                return static_cast<uint16_t>((nax >> 16u) & 0xFu);
            if (field == 5u)
                return static_cast<uint16_t>(nax & 0xFFFFu);
            return m_regs[offset >> 1u];
        }
        switch (reg)
        {
        case 0x19Au:
            return core.attr;
        case 0x1A8u:
            return static_cast<uint16_t>(core.tsa >> 16u);
        case 0x1AAu:
            return static_cast<uint16_t>(core.tsa & 0xFFFFu);
        case 0x1ACu:
        {
            checkIrq(core.tsa);
            const uint16_t value = m_ram[core.tsa];
            core.tsa = (core.tsa + 1u) & kAddressMask;
            return value;
        }
        case 0x1B0u:
            return core.admas;
        case 0x340u:
            return static_cast<uint16_t>(core.endx & 0xFFFFu);
        case 0x342u:
            return static_cast<uint16_t>(core.endx >> 16u);
        case 0x344u:
            return core.statx;
        default:
            return m_regs[offset >> 1u];
        }
    }

    void IopSpu2::writeLocked(uint32_t offset, uint16_t value)
    {
        offset &= RegisterWindow - 2u;
        m_regs[offset >> 1u] = value;
        if (offset >= 0x760u && offset < 0x7B0u)
        {
            writeCoreExtra(m_cores[(offset - 0x760u) / 0x28u], 0x760u + (offset - 0x760u) % 0x28u, value);
            return;
        }
        if (offset == 0x7C2u)
        {
            m_irqInfo = value;
            return;
        }
        if (offset >= 0x7B0u)
            return;
        const int coreIndex = (offset >> 10u) & 1;
        writeCore(m_cores[coreIndex], coreIndex, offset & 0x3FFu, value);
    }

    void IopSpu2::writeCore(Core &core, int index, uint32_t reg, uint16_t value)
    {
        if (traceEnabled() && (reg == 0x19Au || reg == 0x19Cu || reg == 0x19Eu || reg == 0x1B0u ||
                               reg == 0x1A4u || reg == 0x1A6u || reg == 0x198u))
            std::fprintf(stderr, "[spu2] w c%d %03x=%04x\n", index, reg, value);
        if (reg < 0x180u)
        {
            Voice &voice = core.voices[reg >> 4u];
            switch ((reg & 0xFu) >> 1u)
            {
            case 0:
                voice.volL.set(value);
                break;
            case 1:
                voice.volR.set(value);
                break;
            case 2:
                voice.pitch = value;
                break;
            case 3:
                voice.adsr1 = value;
                break;
            case 4:
                voice.adsr2 = value;
                break;
            case 5:
                voice.envelope = value & 0x7FFFu;
                break;
            default:
                break;
            }
            return;
        }
        if (reg >= 0x1C0u && reg < 0x2E0u)
        {
            if (traceEnabled())
                std::fprintf(stderr, "[spu2] w c%d v%02u addr%u=%04x\n", index, (reg - 0x1C0u) / 12u,
                             ((reg - 0x1C0u) % 12u) >> 1u, value);
            Voice &voice = core.voices[(reg - 0x1C0u) / 12u];
            switch (((reg - 0x1C0u) % 12u) >> 1u)
            {
            case 0:
                voice.ssa = setHigh(voice.ssa, value);
                break;
            case 1:
                voice.ssa = setLow(voice.ssa, value & 0xFFF8u);
                break;
            case 2:
                voice.lsa = setHigh(voice.lsa, value);
                voice.lsaWritten = true;
                break;
            case 3:
                voice.lsa = setLow(voice.lsa, value & 0xFFF8u);
                voice.lsaWritten = true;
                break;
            case 4:
                voice.block = setHigh(voice.block, value) & 0xFFFF8u;
                if (voice.phase != EnvPhase::Stopped)
                    startBlock(voice);
                break;
            case 5:
                voice.block = setLow(voice.block, value & 0xFFF8u);
                if (voice.phase != EnvPhase::Stopped)
                    startBlock(voice);
                break;
            default:
                break;
            }
            return;
        }

        switch (reg)
        {
        case 0x180u:
            core.pmon = setVoiceMaskLow(core.pmon, value);
            break;
        case 0x182u:
            core.pmon = setVoiceMaskHigh(core.pmon, value);
            break;
        case 0x184u:
            core.non = setVoiceMaskLow(core.non, value);
            break;
        case 0x186u:
            core.non = setVoiceMaskHigh(core.non, value);
            break;
        case 0x188u:
            core.vmixL = setVoiceMaskLow(core.vmixL, value);
            break;
        case 0x18Au:
            core.vmixL = setVoiceMaskHigh(core.vmixL, value);
            break;
        case 0x18Cu:
            core.vmixEL = setVoiceMaskLow(core.vmixEL, value);
            break;
        case 0x18Eu:
            core.vmixEL = setVoiceMaskHigh(core.vmixEL, value);
            break;
        case 0x190u:
            core.vmixR = setVoiceMaskLow(core.vmixR, value);
            break;
        case 0x192u:
            core.vmixR = setVoiceMaskHigh(core.vmixR, value);
            break;
        case 0x194u:
            core.vmixER = setVoiceMaskLow(core.vmixER, value);
            break;
        case 0x196u:
            core.vmixER = setVoiceMaskHigh(core.vmixER, value);
            break;
        case 0x198u:
            core.mmix = value;
            break;
        case 0x19Au:
            setAttr(core, index, value);
            break;
        case 0x19Cu:
            core.irqa = setHigh(core.irqa, value);
            break;
        case 0x19Eu:
            core.irqa = setLow(core.irqa, value);
            break;
        case 0x1A0u:
            keyOn(core, index, value);
            break;
        case 0x1A2u:
            keyOn(core, index, static_cast<uint32_t>(value & 0xFFu) << 16u);
            break;
        case 0x1A4u:
            keyOff(core, value);
            break;
        case 0x1A6u:
            keyOff(core, static_cast<uint32_t>(value & 0xFFu) << 16u);
            break;
        case 0x1A8u:
            core.tsa = setHigh(core.tsa, value);
            break;
        case 0x1AAu:
            core.tsa = setLow(core.tsa, value);
            break;
        case 0x1ACu:
            if (traceEnabled())
                std::fprintf(stderr, "[spu2] port c%d tsa=%05x val=%04x\n", index, core.tsa, value);
            checkIrq(core.tsa);
            m_ram[core.tsa] = value;
            core.tsa = (core.tsa + 1u) & kAddressMask;
            break;
        case 0x1B0u:
            core.admas = value;
            if ((value & 3u) == 0u && !core.admaData.empty())
            {
                core.admaData.clear();
                core.admaBlock = 0u;
                core.admaSample = 0u;
            }
            break;
        case 0x2E0u:
            core.esa = setHigh(core.esa, value);
            break;
        case 0x2E2u:
            core.esa = setLow(core.esa, value);
            break;
        case 0x33Cu:
            core.eea = (static_cast<uint32_t>(value & 0xFu) << 16u) | 0xFFFFu;
            break;
        case 0x340u:
            core.endx &= 0xFF0000u;
            break;
        case 0x342u:
            core.endx &= 0xFFFFu;
            break;
        default:
            break;
        }
    }

    void IopSpu2::writeCoreExtra(Core &core, uint32_t reg, uint16_t value)
    {
        switch (reg)
        {
        case 0x760u:
            core.mvolL.set(value);
            break;
        case 0x762u:
            core.mvolR.set(value);
            break;
        case 0x764u:
            core.evolL = static_cast<int16_t>(value);
            break;
        case 0x766u:
            core.evolR = static_cast<int16_t>(value);
            break;
        case 0x768u:
            core.avolL = static_cast<int16_t>(value);
            break;
        case 0x76Au:
            core.avolR = static_cast<int16_t>(value);
            break;
        case 0x76Cu:
            core.bvolL = static_cast<int16_t>(value);
            break;
        case 0x76Eu:
            core.bvolR = static_cast<int16_t>(value);
            break;
        default:
            break;
        }
    }

    void IopSpu2::setAttr(Core &core, int index, uint16_t value)
    {
        const uint16_t oldDmaMode = (core.attr >> 4u) & 3u;
        const bool oldIrqEnable = (core.attr & kAttrIrqEnable) != 0u;
        const uint16_t dmaMode = (value >> 4u) & 3u;
        core.attr = value;
        if (dmaMode == 0u && (core.statx & kStatxDmaBusy) == 0u)
            core.statx &= ~kStatxDmaReady;
        else if (oldDmaMode == 0u && dmaMode != 0u)
            core.statx |= kStatxDmaReady;
        if (oldIrqEnable && (value & kAttrIrqEnable) == 0u)
            m_irqInfo &= ~static_cast<uint16_t>(4u << index);
    }

    void IopSpu2::keyOn(Core &core, int coreIndex, uint32_t mask)
    {
        (void)coreIndex;
        core.endx &= ~mask;
        for (uint32_t index = 0; index < 24u; ++index)
        {
            if ((mask & (1u << index)) == 0u)
                continue;
            Voice &voice = core.voices[index];
            if (traceEnabled())
                std::fprintf(stderr, "[spu2] kon c%d v%02u ssa=%05x pitch=%04x adsr=%04x:%04x vol=%04x:%04x\n",
                             coreIndex, index, voice.ssa, voice.pitch, voice.adsr1, voice.adsr2,
                             voice.volL.reg, voice.volR.reg);
            voice.block = voice.ssa & 0xFFFF8u;
            voice.lsaWritten = false;
            voice.phase = EnvPhase::Attack;
            voice.envelope = 0;
            voice.envCounter = 0u;
            voice.counter = 0u;
            voice.prev1 = 0;
            voice.prev2 = 0;
            voice.history.fill(0);
            voice.out = 0;
            startBlock(voice);
        }
    }

    void IopSpu2::keyOff(Core &core, uint32_t mask)
    {
        for (uint32_t index = 0; index < 24u; ++index)
        {
            if ((mask & (1u << index)) == 0u)
                continue;
            Voice &voice = core.voices[index];
            if (voice.phase == EnvPhase::Stopped)
                continue;
            voice.phase = EnvPhase::Release;
            voice.envCounter = 0u;
        }
    }

    void IopSpu2::startBlock(Voice &voice)
    {
        const uint32_t address = voice.block & 0xFFFF8u;
        checkIrqRange(address, 8u);
        const uint16_t header = m_ram[address];
        if (((header >> 8u) & kFlagLoopStart) != 0u && !voice.lsaWritten)
        {
            if (traceEnabled() && voice.lsa != address)
                std::fprintf(stderr, "[spu2] lstart block=%05x flags=%x\n", address, header >> 8u);
            voice.lsa = address;
        }

        const int shift = (header & 0xFu) + 16;
        const uint32_t filter = std::min<uint32_t>((header >> 4u) & 0xFu, 4u);
        const int32_t f0 = kAdpcmFilters[filter][0];
        const int32_t f1 = kAdpcmFilters[filter][1];
        int32_t prev1 = voice.prev1;
        int32_t prev2 = voice.prev2;
        for (uint32_t word = 0; word < 7u; ++word)
        {
            const uint16_t data = m_ram[address + 1u + word];
            for (uint32_t nibble = 0; nibble < 4u; ++nibble)
            {
                const int32_t raw = static_cast<int32_t>(static_cast<uint32_t>(data >> (nibble * 4u)) << 28u) >> shift;
                const int32_t sample = clamp16(raw + ((f0 * prev1 + f1 * prev2 + 32) >> 6));
                voice.decoded[word * 4u + nibble] = static_cast<int16_t>(sample);
                prev2 = prev1;
                prev1 = sample;
            }
        }
        voice.prev1 = prev1;
        voice.prev2 = prev2;
        voice.samplePos = 0u;
    }

    int16_t IopSpu2::nextSample(Core &core, int coreIndex, int voiceIndex)
    {
        (void)coreIndex;
        Voice &voice = core.voices[voiceIndex];
        if (voice.samplePos >= 28u)
        {
            const uint32_t address = voice.block & 0xFFFF8u;
            const uint8_t flags = static_cast<uint8_t>(m_ram[address] >> 8u);
            if ((flags & kFlagLoopEnd) != 0u)
            {
                if (traceEnabled())
                    std::fprintf(stderr, "[spu2] end c%d v%02d block=%05x flags=%x lsa=%05x lsaw=%d\n", coreIndex,
                                 voiceIndex, address, flags, voice.lsa, voice.lsaWritten ? 1 : 0);
                core.endx |= 1u << voiceIndex;
                voice.block = voice.lsa & 0xFFFF8u;
                if ((flags & kFlagLoop) == 0u)
                {
                    voice.phase = EnvPhase::Stopped;
                    voice.envelope = 0;
                }
            }
            else
            {
                voice.block = (address + 8u) & kAddressMask;
            }
            startBlock(voice);
        }
        return voice.decoded[voice.samplePos++];
    }

    void IopSpu2::checkIrq(uint32_t address)
    {
        for (int index = 0; index < 2; ++index)
        {
            const Core &core = m_cores[index];
            if ((core.attr & kAttrIrqEnable) != 0u && core.irqa == address)
                raiseIrq(index);
        }
    }

    void IopSpu2::checkIrqRange(uint32_t start, uint32_t halfwords)
    {
        for (int index = 0; index < 2; ++index)
        {
            const Core &core = m_cores[index];
            if ((core.attr & kAttrIrqEnable) == 0u)
                continue;
            if (((core.irqa - start) & kAddressMask) < halfwords)
                raiseIrq(index);
        }
    }

    void IopSpu2::raiseIrq(int core)
    {
        const uint16_t bit = static_cast<uint16_t>(4u << core);
        if ((m_irqInfo & bit) != 0u)
            return;
        if (traceEnabled())
            std::fprintf(stderr, "[spu2] irq c%d irqa=%05x\n", core, m_cores[core].irqa);
        m_irqInfo |= bit;
        m_irqPending.store(true, std::memory_order_release);
    }

    void IopSpu2::stepEnvelope(Voice &voice)
    {
        bool decrease = false;
        bool exponential = false;
        int shift = 0;
        int step = 0;
        int32_t target = 0;
        switch (voice.phase)
        {
        case EnvPhase::Attack:
            exponential = (voice.adsr1 & 0x8000u) != 0u;
            shift = (voice.adsr1 >> 10u) & 0x1Fu;
            step = 7 - ((voice.adsr1 >> 8u) & 3u);
            target = 0x7FFF;
            break;
        case EnvPhase::Decay:
            decrease = true;
            exponential = true;
            shift = (voice.adsr1 >> 4u) & 0xFu;
            step = -8;
            target = ((voice.adsr1 & 0xFu) + 1) << 11;
            break;
        case EnvPhase::Sustain:
            decrease = (voice.adsr2 & 0x4000u) != 0u;
            exponential = (voice.adsr2 & 0x8000u) != 0u;
            shift = (voice.adsr2 >> 8u) & 0x1Fu;
            step = 7 - ((voice.adsr2 >> 6u) & 3u);
            if (decrease)
                step = ~step;
            break;
        case EnvPhase::Release:
            decrease = true;
            exponential = (voice.adsr2 & 0x20u) != 0u;
            shift = voice.adsr2 & 0x1Fu;
            step = -8;
            break;
        case EnvPhase::Stopped:
            voice.envelope = 0;
            return;
        }

        const EnvelopeStep delta = envelopeStep(decrease, exponential, shift, step, voice.envelope);
        voice.envCounter += std::max<uint32_t>(1u, delta.counterIncrement);
        if (voice.envCounter >= 0x8000u)
        {
            voice.envCounter = 0u;
            voice.envelope = std::clamp<int32_t>(voice.envelope + delta.levelIncrement, 0, 0x7FFF);
        }

        switch (voice.phase)
        {
        case EnvPhase::Attack:
            if (voice.envelope >= target)
                voice.phase = EnvPhase::Decay;
            break;
        case EnvPhase::Decay:
            if (voice.envelope <= target)
                voice.phase = EnvPhase::Sustain;
            break;
        case EnvPhase::Sustain:
            if (voice.envelope == 0)
                voice.phase = EnvPhase::Stopped;
            break;
        case EnvPhase::Release:
            if (voice.envelope <= 0)
            {
                voice.envelope = 0;
                voice.phase = EnvPhase::Stopped;
            }
            break;
        default:
            break;
        }
    }

    int32_t IopSpu2::mixVoice(Core &core, int coreIndex, int voiceIndex, int32_t previousOut, int32_t &left, int32_t &right)
    {
        Voice &voice = core.voices[voiceIndex];
        left = right = 0;
        if (voice.phase == EnvPhase::Stopped)
        {
            voice.out = 0;
            return 0;
        }

        voice.volL.update();
        voice.volR.update();

        int32_t value;
        if ((core.non >> voiceIndex) & 1u)
        {
            value = core.noiseLevel;
        }
        else
        {
            const auto &weights = gaussTable()[(voice.counter >> 4u) & 0xFFu];
            value = ((weights[0] * voice.history[0]) >> 15) +
                    ((weights[1] * voice.history[1]) >> 15) +
                    ((weights[2] * voice.history[2]) >> 15) +
                    ((weights[3] * voice.history[3]) >> 15);
        }

        stepEnvelope(voice);
        value = applyVolume(value, voice.envelope);
        voice.out = value;
        left = applyVolume(value, voice.volL.value);
        right = applyVolume(value, voice.volR.value);

        int32_t pitch = voice.pitch;
        if (voiceIndex > 0 && ((core.pmon >> voiceIndex) & 1u))
            pitch = std::clamp((pitch * (32768 + previousOut)) >> 15, 0, 0x3FFF);
        voice.counter += static_cast<uint32_t>(std::min(pitch, 0x3FFF));
        uint32_t steps = voice.counter >> 12u;
        voice.counter &= 0xFFFu;
        while (steps-- > 0u && voice.phase != EnvPhase::Stopped)
        {
            const int16_t sample = nextSample(core, coreIndex, voiceIndex);
            voice.history[0] = voice.history[1];
            voice.history[1] = voice.history[2];
            voice.history[2] = voice.history[3];
            voice.history[3] = sample;
        }
        return value;
    }

    void IopSpu2::mixSample(int32_t &outLeft, int32_t &outRight)
    {
        int32_t dryL[2] = {};
        int32_t dryR[2] = {};
        int32_t wetL[2] = {};
        int32_t wetR[2] = {};
        for (int coreIndex = 0; coreIndex < 2; ++coreIndex)
        {
            Core &core = m_cores[coreIndex];

            // Noise generator (shared by every voice of the core flagged in NON).
            if (core.non != 0u)
            {
                const uint32_t clock = (core.attr >> 8u) & 0x3Fu;
                int32_t timer = static_cast<int32_t>(core.noiseCounter) - static_cast<int32_t>(4u + (clock & 3u));
                while (timer < 0)
                {
                    const uint32_t level = static_cast<uint16_t>(core.noiseLevel);
                    const uint32_t parity = ((level >> 15u) ^ (level >> 12u) ^ (level >> 11u) ^ (level >> 10u) ^ 1u) & 1u;
                    core.noiseLevel = static_cast<int16_t>(static_cast<uint16_t>((level << 1u) | parity));
                    timer += static_cast<int32_t>(0x20000u >> (clock >> 2u));
                }
                core.noiseCounter = static_cast<uint32_t>(timer);
            }

            int32_t previousOut = 0;
            for (int voiceIndex = 0; voiceIndex < 24; ++voiceIndex)
            {
                int32_t left;
                int32_t right;
                previousOut = mixVoice(core, coreIndex, voiceIndex, previousOut, left, right);
                if ((core.vmixL >> voiceIndex) & 1u)
                    dryL[coreIndex] += left;
                if ((core.vmixR >> voiceIndex) & 1u)
                    dryR[coreIndex] += right;
                if ((core.vmixEL >> voiceIndex) & 1u)
                    wetL[coreIndex] += left;
                if ((core.vmixER >> voiceIndex) & 1u)
                    wetR[coreIndex] += right;
            }
            dryL[coreIndex] = clamp16(dryL[coreIndex]);
            dryR[coreIndex] = clamp16(dryR[coreIndex]);
            wetL[coreIndex] = clamp16(wetL[coreIndex]);
            wetR[coreIndex] = clamp16(wetR[coreIndex]);
            core.mvolL.update();
            core.mvolR.update();
        }

        // Each core: dry path + reverb of the wet path, then master volume.
        // Core 0 has no external input; its output feeds core 1's external input.
        int32_t extL = 0;
        int32_t extR = 0;
        for (int coreIndex = 0; coreIndex < 2; ++coreIndex)
        {
            Core &core = m_cores[coreIndex];
            const uint16_t mmix = coreIndex ? core.mmix : (core.mmix & 0xFF0u);
            int32_t inputL;
            int32_t inputR;
            readAdmaInput(core, coreIndex, inputL, inputR);
            inputL = applyVolume(inputL, core.bvolL);
            inputR = applyVolume(inputR, core.bvolR);
            int32_t left = ((mmix & 0x800u) ? dryL[coreIndex] : 0) + ((mmix & 0x8u) ? extL : 0) +
                           ((mmix & 0x80u) ? inputL : 0);
            int32_t right = ((mmix & 0x400u) ? dryR[coreIndex] : 0) + ((mmix & 0x4u) ? extR : 0) +
                            ((mmix & 0x40u) ? inputR : 0);
            const int32_t wetInL = ((mmix & 0x200u) ? wetL[coreIndex] : 0) + ((mmix & 0x2u) ? extL : 0) +
                                   ((mmix & 0x20u) ? inputL : 0);
            const int32_t wetInR = ((mmix & 0x100u) ? wetR[coreIndex] : 0) + ((mmix & 0x1u) ? extR : 0) +
                                   ((mmix & 0x10u) ? inputR : 0);
            int32_t reverbL = 0;
            int32_t reverbR = 0;
            reverb(core, coreIndex, wetInL, wetInR, reverbL, reverbR);
            left += applyVolume(reverbL, core.evolL);
            right += applyVolume(reverbR, core.evolR);
            left = applyVolume(clamp16(left), core.mvolL.value);
            right = applyVolume(clamp16(right), core.mvolR.value);
            if (coreIndex == 0)
            {
                const Core &next = m_cores[1];
                extL = applyVolume(left, next.avolL);
                extR = applyVolume(right, next.avolR);
            }
            else
            {
                outLeft = clamp16(left);
                outRight = clamp16(right);
            }
        }
        ++m_sampleClock;
    }

    uint32_t IopSpu2::reverbAddress(const Core &core, int index, uint32_t reg) const
    {
        const uint32_t base = (index ? 0x400u : 0u) + reg;
        return (static_cast<uint32_t>(m_regs[base >> 1u] & 0xFu) << 16u) | m_regs[(base >> 1u) + 1u];
    }

    void IopSpu2::reverb(Core &core, int index, int32_t inLeft, int32_t inRight, int32_t &outLeft, int32_t &outRight)
    {
        outLeft = outRight = 0;
        if ((core.attr & kAttrFxEnable) == 0u || core.esa >= core.eea)
            return;

        const uint32_t pos = core.reverbPos;
        core.reverbDown[0][pos] = core.reverbDown[0][pos | 64u] = static_cast<int16_t>(clamp16(inLeft));
        core.reverbDown[1][pos] = core.reverbDown[1][pos | 64u] = static_cast<int16_t>(clamp16(inRight));

        const bool right = (m_sampleClock & 1u) != 0u;
        const int64_t size = static_cast<int64_t>(core.eea) - core.esa + 1;
        const int64_t clock = m_sampleClock >> 1u;
        const auto at = [&](int64_t offset) -> uint32_t
        {
            int64_t x = (clock + offset) % size;
            if (x < 0)
                x += size;
            return static_cast<uint32_t>(core.esa + x) & kAddressMask;
        };
        const auto reg = [&](uint32_t offset) -> int64_t
        { return reverbAddress(core, index, offset); };
        const auto vol = [&](uint32_t offset) -> int32_t
        { return static_cast<int16_t>(m_regs[((index ? 0x788u : 0x760u) + offset) >> 1u]); };
        const auto mem = [&](uint32_t address) -> int32_t
        { return static_cast<int16_t>(m_ram[address]); };
        const auto mul = [](int32_t a, int32_t b) -> int32_t
        { return (a * b) >> 15; };

        const int64_t apf1Size = reg(0x2E4u);
        const int64_t apf2Size = reg(0x2E8u);
        const uint32_t sameSrc = at(reg(right ? 0x308u : 0x304u));
        const uint32_t sameDst = at(reg(right ? 0x2F0u : 0x2ECu));
        const uint32_t samePrev = at(reg(right ? 0x2F0u : 0x2ECu) - 1);
        const uint32_t diffSrc = at(reg(right ? 0x324u : 0x328u));
        const uint32_t diffDst = at(reg(right ? 0x310u : 0x30Cu));
        const uint32_t diffPrev = at(reg(right ? 0x310u : 0x30Cu) - 1);
        const uint32_t comb1 = at(reg(right ? 0x2F8u : 0x2F4u));
        const uint32_t comb2 = at(reg(right ? 0x300u : 0x2FCu));
        const uint32_t comb3 = at(reg(right ? 0x318u : 0x314u));
        const uint32_t comb4 = at(reg(right ? 0x320u : 0x31Cu));
        const int64_t apf1Base = reg(right ? 0x330u : 0x32Cu);
        const int64_t apf2Base = reg(right ? 0x338u : 0x334u);
        const uint32_t apf1Src = at(apf1Base - apf1Size);
        const uint32_t apf1Dst = at(apf1Base);
        const uint32_t apf2Src = at(apf2Base - apf2Size);
        const uint32_t apf2Dst = at(apf2Base);

        for (int other = 0; other < 2; ++other)
        {
            const Core &irqCore = m_cores[other];
            if ((irqCore.attr & kAttrIrqEnable) == 0u || irqCore.irqa < core.esa || irqCore.irqa > core.eea)
                continue;
            for (const uint32_t address : {sameSrc, sameDst, samePrev, diffSrc, diffDst, diffPrev, comb1, comb2,
                                           comb3, comb4, apf1Src, apf1Dst, apf2Src, apf2Dst})
            {
                if (address == irqCore.irqa)
                {
                    raiseIrq(other);
                    break;
                }
            }
        }

        // 39-tap half-band decimation to 24 kHz.
        int32_t input = 0;
        {
            const uint32_t first = (pos - kReverbTaps) & 63u;
            const auto &history = core.reverbDown[right ? 1 : 0];
            for (uint32_t tap = 0; tap < kReverbTaps; ++tap)
                input += history[first + tap] * kReverbFilter[tap];
            input = clamp16(input >> 15);
        }

        const int32_t in = mul(vol(right ? 0x26u : 0x24u), input);
        const int32_t iir = vol(0x14u);
        const int32_t wall = vol(0x1Eu);
        const int32_t same = mul(iir, in + mul(wall, mem(sameSrc)) - mem(samePrev)) + mem(samePrev);
        const int32_t diff = mul(iir, in + mul(wall, mem(diffSrc)) - mem(diffPrev)) + mem(diffPrev);
        int32_t out = mul(vol(0x16u), mem(comb1)) + mul(vol(0x18u), mem(comb2)) +
                      mul(vol(0x1Au), mem(comb3)) + mul(vol(0x1Cu), mem(comb4));
        const int32_t apf1Vol = vol(0x20u);
        const int32_t apf2Vol = vol(0x22u);
        const int32_t apf1 = out - mul(apf1Vol, mem(apf1Src));
        out = mem(apf1Src) + mul(apf1Vol, apf1);
        const int32_t apf2 = out - mul(apf2Vol, mem(apf2Src));
        out = mem(apf2Src) + mul(apf2Vol, apf2);

        m_ram[sameDst] = static_cast<uint16_t>(clamp16(same));
        m_ram[diffDst] = static_cast<uint16_t>(clamp16(diff));
        m_ram[apf1Dst] = static_cast<uint16_t>(clamp16(apf1));
        m_ram[apf2Dst] = static_cast<uint16_t>(clamp16(apf2));
        out = clamp16(out);

        core.reverbUp[right ? 1 : 0][pos] = core.reverbUp[right ? 1 : 0][pos | 64u] = static_cast<int16_t>(out);
        core.reverbUp[right ? 0 : 1][pos] = core.reverbUp[right ? 0 : 1][pos | 64u] = 0;
        core.reverbPos = (pos + 1u) & 63u;

        // Interpolate back to 48 kHz (zero-stuffed, so the filter gain is doubled).
        const uint32_t first = (core.reverbPos - kReverbTaps) & 63u;
        int32_t left = 0;
        int32_t rightOut = 0;
        for (uint32_t tap = 0; tap < kReverbTaps; ++tap)
        {
            const int32_t coefficient = std::clamp<int32_t>(kReverbFilter[tap] * 2, -0x8000, 0x7FFF);
            left += core.reverbUp[0][first + tap] * coefficient;
            rightOut += core.reverbUp[1][first + tap] * coefficient;
        }
        outLeft = clamp16(left >> 15);
        outRight = clamp16(rightOut >> 15);
    }

    void IopSpu2::renderLocked(int16_t *stereo, uint32_t frames)
    {
        for (uint32_t frame = 0; frame < frames; ++frame)
        {
            int32_t left;
            int32_t right;
            mixSample(left, right);
            if (stereo)
            {
                stereo[frame * 2u] = static_cast<int16_t>(left);
                stereo[frame * 2u + 1u] = static_cast<int16_t>(right);
            }
        }
    }

    void IopSpu2::render(int16_t *stereo, uint32_t frames)
    {
        for (uint32_t done = 0; done < frames;)
        {
            const uint32_t count = std::min(kRenderSlice, frames - done);
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                renderLocked(stereo ? stereo + done * 2u : nullptr, count);
            }
            done += count;
        }
    }

    void IopSpu2::advanceIopCycles(uint64_t cycles)
    {
        m_iopCycleCarry += cycles;
        if (m_iopCycleCarry < IopCyclesPerSample)
            return;
        const uint64_t samples = m_iopCycleCarry / IopCyclesPerSample;
        m_iopCycleCarry %= IopCyclesPerSample;
        if (m_hostDriven.load(std::memory_order_relaxed))
        {
            // The host thread mixes, but never ahead of IOP time (renderPaced).
            m_iopSampleClock.fetch_add(samples, std::memory_order_release);
            return;
        }
        // Bound catch-up after long stalls; the voices only need to keep moving.
        render(nullptr, static_cast<uint32_t>(std::min<uint64_t>(samples, SampleRate / 10u)));
    }

    uint32_t IopSpu2::renderPaced(int16_t *stereo, uint32_t maxFrames)
    {
        // The SPU2 runs on the IOP clock: drivers judge voice progress (NAX,
        // ENDX) against IOP timers, so it must never run ahead of the IOP.
        const uint64_t clock = m_iopSampleClock.load(std::memory_order_acquire);
        uint64_t behind = clock - m_renderedSamples;
        if (behind > SampleRate * 2u)
        {
            // Long host stall: resynchronise instead of rendering seconds of audio.
            m_renderedSamples = clock - SampleRate / 10u;
            behind = SampleRate / 10u;
        }
        if (behind > kMaxPacedLag)
        {
            // Keep voice timing within kMaxPacedLag of the IOP; drop that audio.
            const uint32_t skip = static_cast<uint32_t>(behind - kMaxPacedLag);
            render(nullptr, skip);
            m_renderedSamples += skip;
            behind = kMaxPacedLag;
        }
        const uint32_t frames = static_cast<uint32_t>(std::min<uint64_t>(behind, maxFrames));
        render(stereo, frames);
        m_renderedSamples += frames;
        return frames;
    }

    bool IopSpu2::dmaWrite(int core, const uint8_t *source, uint32_t bytes)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        Core &state = m_cores[core & 1];
        const uint32_t halfwords = bytes / 2u;
        state.statx &= ~kStatxDmaReady;
        state.statx |= kStatxDmaBusy;
        if (traceEnabled())
            std::fprintf(stderr, "[spu2] dma-write c%d tsa=%05x bytes=%u admas=%x\n", core & 1, state.tsa, bytes, state.admas);
        // ATV_SPU_DMA_DUMP=dir writes every DMA payload to dir/NNNNNN_c<core>_<tsa>.bin.
        static const char *dumpDir = std::getenv("ATV_SPU_DMA_DUMP");
        if (dumpDir && dumpDir[0] != '\0')
        {
            static uint32_t sequence = 0u;
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%06u_c%d_%05x.bin", dumpDir, sequence++, core & 1, state.tsa);
            if (FILE *file = std::fopen(path, "wb"))
            {
                std::fwrite(source, 1u, bytes, file);
                std::fclose(file);
            }
        }
        if ((state.admas & (1u << (core & 1))) != 0u)
        {
            // PCM input: queue whole 512-byte left/right block pairs for the mixer.
            const size_t samples = (bytes / 1024u) * 512u;
            if (samples == 0u)
                return true;
            const size_t offset = state.admaData.size();
            state.admaData.resize(offset + samples);
            std::memcpy(state.admaData.data() + offset, source, samples * sizeof(int16_t));
            return false;
        }
        checkIrqRange(state.tsa, halfwords);
        uint32_t address = state.tsa;
        for (uint32_t index = 0; index < halfwords; ++index)
        {
            uint16_t value;
            std::memcpy(&value, source + index * 2u, sizeof(value));
            m_ram[address] = value;
            address = (address + 1u) & kAddressMask;
        }
        state.tsa = address;
        return true;
    }

    void IopSpu2::readAdmaInput(Core &core, int index, int32_t &left, int32_t &right)
    {
        left = right = 0;
        if (core.admaData.empty())
            return;
        const size_t base = core.admaBlock * 512u;
        left = core.admaData[base + core.admaSample];
        right = core.admaData[base + 256u + core.admaSample];
        if (++core.admaSample < 256u)
            return;
        core.admaSample = 0u;
        if ((++core.admaBlock) * 512u < core.admaData.size())
            return;
        core.admaData.clear();
        core.admaBlock = 0u;
        m_admaDone.fetch_or(1u << index, std::memory_order_acq_rel);
    }

    void IopSpu2::dmaRead(int core, uint8_t *destination, uint32_t bytes)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        Core &state = m_cores[core & 1];
        const uint32_t halfwords = bytes / 2u;
        state.statx &= ~kStatxDmaReady;
        state.statx |= kStatxDmaBusy;
        if (traceEnabled())
            std::fprintf(stderr, "[spu2] dma-read c%d tsa=%05x bytes=%u\n", core & 1, state.tsa, bytes);
        checkIrqRange(state.tsa, halfwords);
        uint32_t address = state.tsa;
        for (uint32_t index = 0; index < halfwords; ++index)
        {
            const uint16_t value = m_ram[address];
            std::memcpy(destination + index * 2u, &value, sizeof(value));
            address = (address + 1u) & kAddressMask;
        }
        state.tsa = address;
    }

    void IopSpu2::dmaComplete(int core)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        Core &state = m_cores[core & 1];
        if (((state.attr >> 4u) & 3u) != 0u)
            state.statx |= kStatxDmaReady;
        state.statx &= ~kStatxDmaBusy;
    }
}
