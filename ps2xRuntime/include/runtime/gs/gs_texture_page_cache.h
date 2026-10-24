#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace GSMem
{
    // Snapshot of GS local-memory pages. Reads keep seeing the snapshotted
    // page until TextureFlush, matching GS TEXFLUSH behaviour.
    // Direct-mapped by page number: a full-screen texture spans ~5 pages per
    // scanline (4-bit pages are 128 texels wide), which a 4-slot round robin
    // re-copied (8 KB each) on nearly every page change.
    class TexturePageCache
    {
    public:
        static constexpr uint32_t kPageSize = 8192u;
        static constexpr uint32_t kSlots = 64u;

        TexturePageCache() noexcept { Invalidate(); }

        void Invalidate() noexcept
        {
            m_pageBase.fill(UINT32_MAX);
        }

        // byteAddress is the wrapped, swizzled VRAM address. The returned
        // pointer is valid only until the next miss or invalidation that
        // reuses the same slot.
        const uint8_t* Resolve(const uint8_t* vram, uint32_t byteAddress) noexcept
        {
            const uint32_t pageBase = byteAddress & ~(kPageSize - 1u);
            const uint32_t slot = (pageBase / kPageSize) % kSlots;
            if (m_pageBase[slot] != pageBase)
            {
                std::memcpy(m_bytes[slot].data(), vram + pageBase, kPageSize);
                m_pageBase[slot] = pageBase;
            }
            return m_bytes[slot].data() + (byteAddress & (kPageSize - 1u));
        }

    private:
        alignas(64) std::array<std::array<uint8_t, kPageSize>, kSlots> m_bytes{};
        std::array<uint32_t, kSlots> m_pageBase{};
    };
}
