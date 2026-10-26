#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ps2_audio
{
    // The SPU2 follows the IOP clock, which runs at the emulation speed. This
    // WSOLA time stretcher turns that variable-rate stream into real-time
    // output: below full speed audio is stretched (pitch kept) instead of
    // gapping, and the buffered amount steers the playback rate.
    class TimeStretcher
    {
    public:
        TimeStretcher()
        {
            constexpr double kPi = 3.14159265358979323846;
            for (uint32_t i = 0; i < kWindow; ++i)
                m_window[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / kWindow));
            m_tail.assign(kHop * 2u, 0.0f);
        }

        // pull(int16_t *stereo, uint32_t maxFrames) -> frames delivered.
        template <typename Pull>
        void process(int16_t *out, uint32_t frames, Pull &&pull)
        {
            uint32_t done = 0;
            while (done < frames)
            {
                if (m_ready.size() - m_readyPos >= 2u)
                {
                    const size_t available = (m_ready.size() - m_readyPos) / 2u;
                    const uint32_t count = static_cast<uint32_t>(std::min<size_t>(available, frames - done));
                    for (uint32_t i = 0; i < count * 2u; ++i)
                        out[done * 2u + i] = toPcm(m_ready[m_readyPos + i]);
                    m_readyPos += count * 2u;
                    done += count;
                    continue;
                }
                m_ready.clear();
                m_readyPos = 0;
                topUp(pull);
                synthesizeHop();
            }
        }

    private:
        static constexpr uint32_t kWindow = 1024u;     // frames per grain (~21 ms)
        static constexpr uint32_t kHop = kWindow / 2u; // output frames per grain
        static constexpr uint32_t kSearch = 160u;      // alignment search, frames
        static constexpr double kTargetFill = 4096.0;  // buffered frames (~85 ms)

        static int16_t toPcm(float value)
        {
            return static_cast<int16_t>(std::clamp(std::lround(value), -32768L, 32767L));
        }

        size_t inputFrames() const { return m_input.size() / 2u; }

        template <typename Pull>
        void topUp(Pull &pull)
        {
            const double fill = static_cast<double>(inputFrames()) - m_nominal;
            if (fill >= kTargetFill * 2.0)
                return;
            const uint32_t want = static_cast<uint32_t>(kTargetFill * 2.0 - fill);
            m_scratch.resize(static_cast<size_t>(want) * 2u);
            const uint32_t got = pull(m_scratch.data(), want);
            const size_t base = m_input.size();
            m_input.resize(base + static_cast<size_t>(got) * 2u);
            for (size_t i = 0; i < static_cast<size_t>(got) * 2u; ++i)
                m_input[base + i] = m_scratch[i];
        }

        void synthesizeHop()
        {
            const int64_t available = static_cast<int64_t>(inputFrames());
            const double fill = static_cast<double>(available) - m_nominal;
            // Steer toward the target fill: below full speed the fill drops
            // and the input is consumed more slowly (time stretch). When the
            // IOP pauses the rate reaches zero and the current sound is held.
            const double wanted = std::clamp((fill - kWindow) / (kTargetFill - kWindow), 0.0, 2.0);
            m_rate += (wanted - m_rate) * 0.15;
            if (fill > kTargetFill * 6.0)
                m_nominal = static_cast<double>(available) - kTargetFill; // drop a backlog

            m_ready.assign(kHop * 2u, 0.0f);
            if (m_previous < 0)
            {
                if (available >= static_cast<int64_t>(kWindow + kSearch * 2u))
                {
                    m_previous = kSearch;
                    m_nominal = kSearch;
                }
                return;
            }
            const int64_t nominal = static_cast<int64_t>(m_nominal);
            const int64_t natural = m_previous + kHop;
            const int64_t lowest = std::max<int64_t>(0, nominal - kSearch);
            const int64_t highest = std::min<int64_t>(nominal + kSearch, available - kWindow);
            if (highest < lowest || natural + kHop > available)
            {
                // Nothing usable at all (long IOP stall): fade out the tail.
                for (uint32_t i = 0; i < kHop * 2u; ++i)
                    m_ready[i] = m_tail[i];
                std::fill(m_tail.begin(), m_tail.end(), 0.0f);
                return;
            }

            const int64_t start = bestAlignment(natural, lowest, highest);
            for (uint32_t i = 0; i < kHop; ++i)
            {
                const float w = m_window[i];
                const size_t at = static_cast<size_t>(start + i) * 2u;
                m_ready[i * 2u] = m_tail[i * 2u] + w * m_input[at];
                m_ready[i * 2u + 1u] = m_tail[i * 2u + 1u] + w * m_input[at + 1u];
            }
            for (uint32_t i = 0; i < kHop; ++i)
            {
                const float w = m_window[kHop + i];
                const size_t at = static_cast<size_t>(start + kHop + i) * 2u;
                m_tail[i * 2u] = w * m_input[at];
                m_tail[i * 2u + 1u] = w * m_input[at + 1u];
            }
            m_previous = start;
            m_nominal = std::min(m_nominal + kHop * m_rate,
                                 static_cast<double>(available - static_cast<int64_t>(kWindow)));
            discardConsumed();
        }

        // Candidate start whose first half best continues the previous grain
        // (normalised cross-correlation on a decimated mono mix).
        int64_t bestAlignment(int64_t natural, int64_t lowest, int64_t highest) const
        {
            int64_t best = std::clamp(natural, lowest, highest);
            double bestScore = -1e300;
            for (int64_t candidate = lowest; candidate <= highest; candidate += 2)
            {
                double correlation = 0.0;
                double energy = 1e-3;
                for (uint32_t i = 0; i < kHop; i += 4u)
                {
                    const size_t a = static_cast<size_t>(candidate + i) * 2u;
                    const size_t b = static_cast<size_t>(natural + i) * 2u;
                    const double x = m_input[a] + m_input[a + 1u];
                    const double y = m_input[b] + m_input[b + 1u];
                    correlation += x * y;
                    energy += x * x;
                }
                const double score = correlation / std::sqrt(energy);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = candidate;
                }
            }
            return best;
        }

        void discardConsumed()
        {
            const int64_t keep = std::min<int64_t>(m_previous, static_cast<int64_t>(m_nominal) - kSearch);
            if (keep < static_cast<int64_t>(kWindow) * 8)
                return;
            m_input.erase(m_input.begin(), m_input.begin() + keep * 2);
            m_previous -= keep;
            m_nominal -= static_cast<double>(keep);
        }

        float m_window[kWindow];
        std::vector<float> m_input;  // interleaved stereo, not yet consumed
        std::vector<float> m_tail;   // second half of the previous grain
        std::vector<float> m_ready;  // synthesized output not yet delivered
        std::vector<int16_t> m_scratch;
        size_t m_readyPos = 0;
        int64_t m_previous = -1;     // start of the previous grain in m_input
        double m_nominal = 0.0;      // nominal read position in m_input
        double m_rate = 1.0;
    };
}
