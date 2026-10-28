#include "gs_texture_replace.h"

#include "runtime/gs/ps2_gs_memory.h"

#define XXH_INLINE_ALL
#include "xxhash.h" // xxHash 0.8.3 (BSD 2-Clause), the XXH3 PCSX2 hashes textures with

#include "raylib.h" // LoadImage / ExportImage (stb_image)

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace GSTexReplace
{
    namespace
    {
        namespace fs = std::filesystem;

        struct Config
        {
            std::vector<fs::path> packs;
            fs::path dump;
            size_t budgetBytes = size_t(2048) << 20;
            uint32_t maxScale = 16;
            bool debug = false;
        };

        const Config &config()
        {
            static const Config c = []
            {
                Config out;
                if (const char *v = std::getenv("ATV2_TEXTURE_PACK"); v && *v)
                {
                    std::string all(v);
                    for (size_t pos = 0; pos <= all.size();)
                    {
                        size_t end = all.find(';', pos);
                        if (end == std::string::npos)
                            end = all.size();
                        if (end > pos)
                            out.packs.emplace_back(all.substr(pos, end - pos));
                        pos = end + 1;
                    }
                }
                if (const char *v = std::getenv("ATV2_TEXTURE_DUMP"); v && *v)
                    out.dump = fs::path(std::string(v));
                if (const char *v = std::getenv("ATV2_TEXTURE_PACK_MB"))
                    out.budgetBytes = size_t(std::max(64, std::atoi(v))) << 20;
                if (const char *v = std::getenv("ATV2_TEXTURE_PACK_SCALE"))
                    out.maxScale = static_cast<uint32_t>(std::clamp(std::atoi(v), 1, 16));
                out.debug = std::getenv("ATV2_TEXTURE_PACK_DEBUG") != nullptr;
                return out;
            }();
            return c;
        }
    }

    const bool g_active = !config().packs.empty() || !config().dump.empty();

    namespace
    {
        // ---- PCSX2 texture names -------------------------------------------------

        // TextureName in GSTextureReplacements.cpp, base level only.
        struct Name
        {
            uint64_t tex0Hash = 0, clutHash = 0;
            uint32_t regionW = 0, regionH = 0;
            uint32_t bits = 0; // PSM:6 TW:4 TH:4 unused:1 TA0:8 AEM:1 TA1:8

            bool operator==(const Name &o) const
            {
                return tex0Hash == o.tex0Hash && clutHash == o.clutHash && regionW == o.regionW && regionH == o.regionH &&
                       bits == o.bits;
            }
        };

        struct NameHash
        {
            size_t operator()(const Name &n) const
            {
                uint64_t h = n.tex0Hash ^ (n.clutHash * 0x9E3779B97F4A7C15ull);
                h ^= (static_cast<uint64_t>(n.regionW) << 48) ^ (static_cast<uint64_t>(n.regionH) << 32) ^ n.bits;
                return static_cast<size_t>(h ^ (h >> 29));
            }
        };

        bool isPalettePsm(uint32_t psm)
        {
            return psm == GS_PSM_T8 || psm == GS_PSM_T4 || psm == GS_PSM_T8H || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
        }

        std::string fileName(const Name &n)
        {
            char buf[96];
            const unsigned long long tex = n.tex0Hash, clut = n.clutHash;
            const bool palette = isPalettePsm(n.bits & 0x3Fu);
            if (n.regionW != 0u || n.regionH != 0u)
            {
                if (palette)
                    std::snprintf(buf, sizeof(buf), "%llx-%llx-r%ux%u-%08x.png", tex, clut, n.regionW, n.regionH, n.bits);
                else
                    std::snprintf(buf, sizeof(buf), "%llx-r%ux%u-%08x.png", tex, n.regionW, n.regionH, n.bits);
            }
            else if (palette)
                std::snprintf(buf, sizeof(buf), "%llx-%llx-%08x.png", tex, clut, n.bits);
            else
                std::snprintf(buf, sizeof(buf), "%llx-%08x.png", tex, n.bits);
            return buf;
        }

        // ParseReplacementName in GSTextureReplacements.cpp, same order of formats.
        bool parseName(const std::string &file, Name &out)
        {
            unsigned long long tex = 0, clut = 0, oldRegion = 0;
            unsigned rw = 0, rh = 0, bits = 0;
            char dot = 0;
            const char *s = file.c_str();
            const auto done = [&](uint64_t clutHash, uint32_t w, uint32_t h)
            {
                out.tex0Hash = tex;
                out.clutHash = clutHash;
                out.regionW = w;
                out.regionH = h;
                out.bits = bits & ~(1u << 14); // RemoveUnusedBits (old TCC bit)
                return true;
            };
            // Old names carry SourceRegion bits: u16 minx, maxx, miny, maxy.
            const auto oldW = [&] { return static_cast<uint32_t>(static_cast<int16_t>(oldRegion >> 16) - static_cast<int16_t>(oldRegion)); };
            const auto oldH = [&] { return static_cast<uint32_t>(static_cast<int16_t>(oldRegion >> 48) - static_cast<int16_t>(oldRegion >> 32)); };
            if (std::sscanf(s, "%llx-%llx-r%ux%u-%08x%c", &tex, &clut, &rw, &rh, &bits, &dot) == 6 && dot == '.')
                return done(clut, rw, rh);
            if (std::sscanf(s, "%llx-r%ux%u-%08x%c", &tex, &rw, &rh, &bits, &dot) == 5 && dot == '.')
                return done(0, rw, rh);
            if (std::sscanf(s, "%llx-%llx-r%llx-%08x%c", &tex, &clut, &oldRegion, &bits, &dot) == 5 && dot == '.')
                return done(clut, oldW(), oldH());
            if (std::sscanf(s, "%llx-r%llx-%08x%c", &tex, &oldRegion, &bits, &dot) == 4 && dot == '.')
                return done(0, oldW(), oldH());
            if (std::sscanf(s, "%llx-%llx-%08x%c", &tex, &clut, &bits, &dot) == 4 && dot == '.')
                return done(clut, 0, 0);
            if (std::sscanf(s, "%llx-%08x%c", &tex, &bits, &dot) == 3 && dot == '.')
                return done(0, 0, 0);
            return false;
        }

        // ---- GS memory layout, as GSLocalMemory::m_psm sees it ---------------------

        constexpr uint32_t kPageBytes = 8192u;
        constexpr uint32_t kPages = 512u; // 4 MB

        constexpr uint8_t kBlock32[4 * 8] = {0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23,
                                             8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31};
        constexpr uint8_t kBlock16[8 * 4] = {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15,
                                             16, 18, 24, 26, 17, 19, 25, 27, 20, 22, 28, 30, 21, 23, 29, 31};
        constexpr uint8_t kBlock16S[8 * 4] = {0, 2, 16, 18, 1, 3, 17, 19, 8, 10, 24, 26, 9, 11, 25, 27,
                                              4, 6, 20, 22, 5, 7, 21, 23, 12, 14, 28, 30, 13, 15, 29, 31};

        enum class Fmt : uint8_t
        {
            C32,
            C24,
            C16,
            Pal
        };

        using ReadFn = decltype(&GSMem::ReadCT32);

        struct PsmInfo
        {
            int bsx = 8, bsy = 8;   // block size in texels (psm_t::bs)
            int cols = 8, rows = 4; // blocks per page
            bool widePage = false;  // 128-texel pages: TBW counts half pages
            bool fullMask = true;   // psm_t::fmsk == 0xFFFFFFFF
            uint32_t pal = 0;       // CLUT entries
            Fmt fmt = Fmt::C32;
            const uint8_t *table = kBlock32;
            ReadFn read = &GSMem::ReadCT32;
        };

        bool psmInfo(uint32_t psm, PsmInfo &p)
        {
            p = PsmInfo{};
            switch (psm)
            {
            case GS_PSM_CT32:
                return true;
            case GS_PSM_CT24:
                p.fullMask = false;
                p.fmt = Fmt::C24;
                p.read = &GSMem::ReadCT24;
                return true;
            case GS_PSM_CT16:
            case GS_PSM_CT16S:
                p.bsx = 16;
                p.cols = 4;
                p.rows = 8;
                p.fullMask = false;
                p.fmt = Fmt::C16;
                p.table = psm == GS_PSM_CT16 ? kBlock16 : kBlock16S;
                p.read = psm == GS_PSM_CT16 ? &GSMem::ReadCT16 : &GSMem::ReadCT16S;
                return true;
            case GS_PSM_T8:
                p.bsx = 16;
                p.bsy = 16;
                p.widePage = true;
                p.pal = 256;
                p.fmt = Fmt::Pal;
                p.read = &GSMem::ReadP8;
                return true;
            case GS_PSM_T4:
                p.bsx = 32;
                p.bsy = 16;
                p.cols = 4;
                p.rows = 8;
                p.widePage = true;
                p.pal = 16;
                p.fmt = Fmt::Pal;
                p.table = kBlock16;
                p.read = &GSMem::ReadP4;
                return true;
            case GS_PSM_T8H:
            case GS_PSM_T4HL:
            case GS_PSM_T4HH:
                p.fullMask = false;
                p.pal = psm == GS_PSM_T8H ? 256u : 16u;
                p.fmt = Fmt::Pal;
                p.read = psm == GS_PSM_T8H ? &GSMem::ReadP8H : (psm == GS_PSM_T4HL ? &GSMem::ReadP4HL : &GSMem::ReadP4HH);
                return true;
            default:
                return false; // depth formats: no replacements
            }
        }

        // GSOffset::bn for the block at block coordinates (bx, by).
        uint32_t blockNumber(const PsmInfo &p, uint32_t bp, uint32_t bw, int bx, int by)
        {
            const uint32_t pagesWide = p.widePage ? (bw >> 1) : bw;
            const uint32_t page = static_cast<uint32_t>(by / p.rows) * pagesWide + static_cast<uint32_t>(bx / p.cols);
            return (bp + page * 32u + p.table[(by % p.rows) * p.cols + (bx % p.cols)]) & 0x3FFFu;
        }

        uint32_t expand16(uint32_t c, const GSTexaReg &texa)
        {
            const uint32_t rgb = ((c & 0x1Fu) << 3) | (((c >> 5) & 0x1Fu) << 11) | (((c >> 10) & 0x1Fu) << 19);
            uint32_t a = (c & 0x8000u) ? texa.ta1 : texa.ta0;
            if (texa.aem && c == 0u)
                a = 0u;
            return rgb | (a << 24);
        }

        uint32_t expand24(uint32_t c, const GSTexaReg &texa)
        {
            const uint32_t rgb = c & 0x00FFFFFFu;
            return rgb | ((texa.aem && rgb == 0u) ? 0u : (static_cast<uint32_t>(texa.ta0) << 24));
        }

        // GSClut::Read32: the 32-bit palette PCSX2 hashes and expands textures with.
        bool readClut32(const GSTex0Reg &tex0, const GSTexaReg &texa, const uint16_t *clut, uint32_t (&out)[256], uint32_t &count)
        {
            const bool eight = tex0.psm == GS_PSM_T8 || tex0.psm == GS_PSM_T8H;
            count = eight ? 256u : 16u;
            if (tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24)
            {
                const uint32_t offset = (static_cast<uint32_t>(tex0.csa) & 15u) << 4;
                for (uint32_t group = 0; group < count; group += 16u)
                {
                    // T8 clamps the CSA offset at the last 16 entries instead of wrapping.
                    const uint32_t base = eight ? std::min(group + offset, 240u) : offset;
                    for (uint32_t j = 0; j < 16u; ++j)
                        out[group + j] = static_cast<uint32_t>(clut[base + j]) | (static_cast<uint32_t>(clut[base + j + 256u]) << 16);
                }
                return true;
            }
            if (tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S)
            {
                const uint32_t base = static_cast<uint32_t>(tex0.csa) << 4;
                for (uint32_t j = 0; j < count; ++j)
                    out[j] = expand16(clut[(base + j) & 0x1FFu], texa);
                return true;
            }
            return false;
        }

        struct Region // GSTextureCache::SourceRegion
        {
            int minX = 0, maxX = 0, minY = 0, maxY = 0;
            bool hasX() const { return maxX != 0 || minX != 0; }
            bool hasY() const { return maxY != 0 || minY != 0; }
        };

        // SourceRegion::Create(TEX0, CLAMP).
        Region makeRegion(const GSTex0Reg &tex0, uint64_t clamp)
        {
            Region r;
            const uint32_t wms = clamp & 3u, wmt = (clamp >> 2) & 3u;
            const uint32_t minu = (clamp >> 4) & 0x3FFu, maxu = (clamp >> 14) & 0x3FFu;
            const uint32_t minv = (clamp >> 24) & 0x3FFu, maxv = (clamp >> 34) & 0x3FFu;
            const uint32_t tw = 1u << (tex0.tw & 0xFu), th = 1u << (tex0.th & 0xFu);
            if (wms == 2u && maxu >= minu)
            {
                const uint32_t rw = maxu - minu + 1u;
                if (rw < tw || maxu >= tw)
                {
                    r.minX = static_cast<int>(minu);
                    r.maxX = static_cast<int>(maxu + 1u);
                }
            }
            else if (wms == 3u && minu != 0u)
            {
                const uint32_t rw = ((minu | maxu) - maxu) + 1u;
                if (rw < tw || (maxu != 0u && rw <= tw))
                {
                    r.minX = static_cast<int>(maxu);
                    r.maxX = static_cast<int>((minu | maxu) + 1u);
                }
            }
            if (wmt == 2u && maxv >= minv)
            {
                const uint32_t rh = maxv - minv + 1u;
                if (rh < th || maxv >= th)
                {
                    r.minY = static_cast<int>(minv);
                    r.maxY = static_cast<int>(maxv + 1u);
                }
            }
            else if (wmt == 3u && minv != 0u)
            {
                const uint32_t rh = ((minv | maxv) - maxv) + 1u;
                if (rh < th || (maxv != 0u && rh <= th))
                {
                    r.minY = static_cast<int>(maxv);
                    r.maxY = static_cast<int>((minv | maxv) + 1u);
                }
            }
            return r;
        }

        struct Rect
        {
            int l, t, r, b;
        };

        struct Texture
        {
            PsmInfo p;
            Region region;
            int tw = 0, th = 0; // hashed size: the region, or 1<<TW x 1<<TH
            Rect rect{}, blocks{};
        };

        bool describe(const GSDrawState &state, Texture &t)
        {
            const auto &tex0 = state.context.tex0;
            if (!psmInfo(tex0.psm, t.p))
                return false;
            t.region = makeRegion(tex0, state.context.clamp);
            if ((tex0.tw > 10u && !t.region.hasX()) || (tex0.th > 10u && !t.region.hasY()))
                return false;
            t.tw = t.region.hasX() ? t.region.maxX - t.region.minX : (1 << tex0.tw);
            t.th = t.region.hasY() ? t.region.maxY - t.region.minY : (1 << tex0.th);
            t.rect = {t.region.hasX() ? t.region.minX : 0, t.region.hasY() ? t.region.minY : 0,
                      t.region.hasX() ? t.region.maxX : t.tw, t.region.hasY() ? t.region.maxY : t.th};
            const auto down = [](int v, int a) { return v / a * a; };
            const auto up = [](int v, int a) { return (v + a - 1) / a * a; };
            t.blocks = {down(t.rect.l, t.p.bsx), down(t.rect.t, t.p.bsy), up(t.rect.r, t.p.bsx), up(t.rect.b, t.p.bsy)};
            return true;
        }

        uint32_t readTexel(const Texture &t, const GSTex0Reg &tex0, const uint8_t *vram, int x, int y)
        {
            return t.p.read(const_cast<uint8_t *>(vram), tex0.tbp0, tex0.tbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y));
        }

        // VRAM pages the texture's blocks lie in.
        void pageRange(const Texture &t, const GSTex0Reg &tex0, uint32_t &firstPage, uint32_t &lastPage)
        {
            firstPage = kPages;
            lastPage = 0;
            for (int by = t.blocks.t / t.p.bsy; by < t.blocks.b / t.p.bsy; ++by)
                for (int bx = t.blocks.l / t.p.bsx; bx < t.blocks.r / t.p.bsx; ++bx)
                {
                    const uint32_t page = blockNumber(t.p, tex0.tbp0, tex0.tbw, bx, by) >> 5;
                    firstPage = std::min(firstPage, page);
                    lastPage = std::max(lastPage, page);
                }
        }

        bool hashesBlocks(const Texture &t)
        {
            return !(t.tw < t.p.bsx || t.th < t.p.bsy || !t.p.fullMask || t.region.maxX > 0 || t.region.minY > 0);
        }

        // XXH3 of the texture's raw 256-byte blocks: PCSX2's hash itself when hashesBlocks(),
        // otherwise a cheap check that a re-upload brought the same bytes back.
        uint64_t hashBlocks(const Texture &t, const GSTex0Reg &tex0, const uint8_t *vram)
        {
            XXH3_state_t st;
            XXH3_64bits_reset(&st);
            for (int by = t.blocks.t / t.p.bsy; by < t.blocks.b / t.p.bsy; ++by)
                for (int bx = t.blocks.l / t.p.bsx; bx < t.blocks.r / t.p.bsx; ++bx)
                    XXH3_64bits_update(&st, vram + static_cast<size_t>(blockNumber(t.p, tex0.tbp0, tex0.tbw, bx, by)) * 256u, 256u);
            return XXH3_64bits_digest(&st);
        }

        // HashTextureLevel in GSTextureCache.cpp.
        uint64_t hashTexture(const Texture &t, const GSDrawState &state, const uint8_t *vram)
        {
            const auto &tex0 = state.context.tex0;
            if (hashesBlocks(t))
                return hashBlocks(t, tex0, vram);
            XXH3_state_t st;
            XXH3_64bits_reset(&st);
            {
                // Expanded: 8-bit indices or 32-bit colours with TEXA applied.
                thread_local std::vector<uint8_t> buffer;
                const bool palette = t.p.pal > 0u;
                const uint32_t bpp = palette ? 1u : 4u;
                const uint32_t pitch = (static_cast<uint32_t>(t.blocks.r) * bpp + 31u) & ~31u;
                buffer.resize(static_cast<size_t>(pitch) * static_cast<uint32_t>(t.blocks.b - t.blocks.t) + 32u);
                for (int y = t.blocks.t; y < t.blocks.b; ++y)
                {
                    uint8_t *row = buffer.data() + static_cast<size_t>(y - t.blocks.t) * pitch;
                    for (int x = t.blocks.l; x < t.blocks.r; ++x)
                    {
                        const uint32_t raw = readTexel(t, tex0, vram, x, y);
                        if (palette)
                        {
                            row[x - t.blocks.l] = static_cast<uint8_t>(raw);
                            continue;
                        }
                        uint32_t c = raw;
                        if (t.p.fmt == Fmt::C24)
                            c = expand24(raw, state.texa);
                        else if (t.p.fmt == Fmt::C16)
                            c = expand16(raw & 0xFFFFu, state.texa);
                        std::memcpy(row + static_cast<size_t>(x - t.blocks.l) * 4u, &c, 4u);
                    }
                }
                // PCSX2 offsets the start by (left - block left) bytes even for 32-bit texels.
                const uint8_t *ptr = buffer.data() + static_cast<size_t>(pitch) * static_cast<uint32_t>(t.rect.t - t.blocks.t) +
                                     static_cast<uint32_t>(t.rect.l - t.blocks.l);
                const size_t rowSize = static_cast<size_t>(t.tw) * bpp;
                for (int y = 0; y < t.th; ++y, ptr += pitch)
                    XXH3_64bits_update(&st, ptr, rowSize);
            }
            return XXH3_64bits_digest(&st);
        }

        uint32_t nameBits(const GSDrawState &state, const PsmInfo &p)
        {
            const auto &tex0 = state.context.tex0;
            uint32_t bits = (tex0.psm & 0x3Fu) | ((tex0.tw & 0xFu) << 6) | ((tex0.th & 0xFu) << 10);
            if (p.fmt == Fmt::C24 || p.fmt == Fmt::C16)
                bits |= (static_cast<uint32_t>(state.texa.ta0) << 15) | ((state.texa.aem ? 1u : 0u) << 23) |
                        (static_cast<uint32_t>(state.texa.ta1) << 24);
            return bits;
        }

        // ---- pack index, loader, dumper ---------------------------------------------

        struct CacheEntry
        {
            std::shared_ptr<const Image> image;
            uint64_t lastUse = 0;
            size_t bytes = 0;
            bool queued = false, failed = false;
        };

        struct DumpJob
        {
            fs::path path;
            int w = 0, h = 0;
            std::vector<uint32_t> rgba;
        };

        std::mutex g_mutex;
        std::condition_variable g_cv;
        bool g_workerStarted = false;
        std::atomic<bool> g_indexed{false};
        std::atomic<uint64_t> g_loadGen{0}; // bumps when an image finishes loading
        std::unordered_map<Name, fs::path, NameHash> g_files;
        std::unordered_map<Name, CacheEntry, NameHash> g_cache;
        size_t g_cacheBytes = 0;
        uint64_t g_useClock = 0;
        std::deque<Name> g_loads;
        std::deque<DumpJob> g_dumps;
        std::unordered_set<Name, NameHash> g_dumped;

        std::atomic<uint64_t> g_statHashes{0}, g_statHits{0}, g_statLoads{0}, g_statDumps{0}, g_statHashNs{0}, g_statFuzzy{0};
        std::atomic<uint64_t> g_statFuzzyTries{0}, g_statFuzzyNs{0}, g_statLoadNs{0}, g_statSlowPath{0};

        // <pack>/fuzzy_index.bin (convert_dds_pack.py): 16x16 RGB box thumbnails of the pack's
        // direct-colour textures. IPU-decoded textures (the race's terrain maps) differ from
        // PCSX2's in their low bits, so their hashes never match; the thumbnail finds them.
        constexpr int kThumb = 16;
        using Thumb = std::array<uint8_t, kThumb * kThumb * 3>;

        struct FuzzyEntry
        {
            Name name;
            Thumb thumb;
        };

        std::unordered_map<uint32_t, std::vector<FuzzyEntry>> g_fuzzy; // by Name::bits; read-only once indexed

        void loadFuzzyIndex(const fs::path &pack, std::unordered_map<uint32_t, std::vector<FuzzyEntry>> &out)
        {
            std::FILE *fh = std::fopen((pack / "fuzzy_index.bin").string().c_str(), "rb");
            if (!fh)
                return;
            char magic[8] = {};
            uint32_t count = 0, size = 0;
            if (std::fread(magic, 1, 8, fh) == 8 && std::memcmp(magic, "ATV2FZ1", 8) == 0 && std::fread(&count, 4, 1, fh) == 1 &&
                std::fread(&size, 4, 1, fh) == 1 && size == kThumb)
            {
                for (uint32_t i = 0; i < count; ++i)
                {
                    uint16_t len = 0;
                    char raw[96] = {};
                    FuzzyEntry e;
                    if (std::fread(&len, 2, 1, fh) != 1 || len >= sizeof(raw) - 4 || std::fread(raw, 1, len, fh) != len ||
                        std::fread(e.thumb.data(), 1, e.thumb.size(), fh) != e.thumb.size())
                        break;
                    std::memcpy(raw + len, ".png", 5);
                    if (parseName(raw, e.name))
                        out[e.name.bits].push_back(e);
                }
            }
            std::fclose(fh);
        }

        void buildIndex()
        {
            std::unordered_map<Name, fs::path, NameHash> files;
            std::unordered_map<uint32_t, std::vector<FuzzyEntry>> fuzzy;
            for (const fs::path &pack : config().packs) // earlier packs win
            {
                const auto start = std::chrono::steady_clock::now();
                fs::path root = pack;
                std::error_code ec;
                if (fs::is_directory(root / "replacements", ec))
                    root /= "replacements";
                size_t added = 0, skipped = 0;
                for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end; it != end;
                     it.increment(ec))
                {
                    if (ec)
                        break;
                    if (!it->is_regular_file(ec))
                        continue;
                    std::string ext = it->path().extension().string();
                    for (char &c : ext)
                        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    Name name;
                    if (ext != ".png" || !parseName(it->path().filename().string(), name))
                    {
                        ++skipped;
                        continue;
                    }
                    added += files.emplace(name, it->path()).second ? 1u : 0u;
                }
                size_t thumbs = 0;
                for (const auto &kv : fuzzy)
                    thumbs -= kv.second.size();
                loadFuzzyIndex(pack, fuzzy);
                for (const auto &kv : fuzzy)
                    thumbs += kv.second.size();
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                std::fprintf(stderr, "[tex-pack] %zu replacement PNGs from %s (%zu other files skipped, %zu thumbnails, %.0f ms)\n",
                             added, root.string().c_str(), skipped, thumbs, ms);
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            g_files.swap(files);
            g_fuzzy.swap(fuzzy);
            g_indexed = true;
            g_loadGen.fetch_add(1, std::memory_order_release);
        }

        std::shared_ptr<const Image> loadImage(const fs::path &path, const Name &name, size_t &bytes)
        {
            ::Image src = ::LoadImage(path.string().c_str());
            if (src.data == nullptr || src.width <= 0 || src.height <= 0)
                return nullptr;
            if (src.format != PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
                ::ImageFormat(&src, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
            auto img = std::make_shared<Image>();
            img->regionW = name.regionW;
            img->regionH = name.regionH;
            const uint32_t w = static_cast<uint32_t>(src.width), h = static_cast<uint32_t>(src.height);
            size_t total = 0;
            uint32_t sizes[16][2];
            uint32_t count = 0;
            for (uint32_t lw = w, lh = h; count < 16u;)
            {
                sizes[count][0] = lw;
                sizes[count][1] = lh;
                total += static_cast<size_t>(lw) * lh;
                ++count;
                if (lw == 1u && lh == 1u)
                    break;
                lw = std::max(1u, lw >> 1);
                lh = std::max(1u, lh >> 1);
            }
            img->storage.resize(total);
            std::memcpy(img->storage.data(), src.data, static_cast<size_t>(w) * h * 4u);
            ::UnloadImage(src);
            size_t offset = 0;
            for (uint32_t l = 0; l < count; ++l)
            {
                const uint32_t lw = sizes[l][0], lh = sizes[l][1];
                uint32_t *dst = img->storage.data() + offset;
                if (l > 0)
                {
                    // 2x2 box filter of the level above (edge texels repeat on odd sizes).
                    const Level &up = img->levels[l - 1];
                    for (uint32_t y = 0; y < lh; ++y)
                        for (uint32_t x = 0; x < lw; ++x)
                        {
                            const uint32_t x0 = std::min(2u * x, up.w - 1u), x1 = std::min(2u * x + 1u, up.w - 1u);
                            const uint32_t y0 = std::min(2u * y, up.h - 1u), y1 = std::min(2u * y + 1u, up.h - 1u);
                            const uint32_t a = up.texels[y0 * up.w + x0], b = up.texels[y0 * up.w + x1];
                            const uint32_t c = up.texels[y1 * up.w + x0], d = up.texels[y1 * up.w + x1];
                            uint32_t out = 0u;
                            for (int s = 0; s < 32; s += 8)
                                out |= ((((a >> s) & 0xFFu) + ((b >> s) & 0xFFu) + ((c >> s) & 0xFFu) + ((d >> s) & 0xFFu) + 2u) >> 2) << s;
                            dst[y * lw + x] = out;
                        }
                }
                img->levels[l] = {lw, lh, dst};
                offset += static_cast<size_t>(lw) * lh;
            }
            img->levelCount = count;
            bytes = total * 4u;
            return img;
        }

        void evictLocked()
        {
            while (g_cacheBytes > config().budgetBytes)
            {
                CacheEntry *oldest = nullptr;
                for (auto &kv : g_cache)
                    if (kv.second.image && (!oldest || kv.second.lastUse < oldest->lastUse))
                        oldest = &kv.second;
                if (!oldest)
                    return;
                g_cacheBytes -= oldest->bytes;
                oldest->image.reset(); // queued draws holding it keep it alive
                oldest->bytes = 0;
                g_loadGen.fetch_add(1, std::memory_order_release); // memos look it up again
            }
        }

        void workerMain()
        {
            if (!config().packs.empty())
                buildIndex();
            if (!config().dump.empty())
            {
                std::error_code ec;
                fs::create_directories(config().dump, ec);
            }
            for (;;)
            {
                Name load{};
                bool haveLoad = false;
                DumpJob dump;
                fs::path path;
                {
                    std::unique_lock<std::mutex> lock(g_mutex);
                    g_cv.wait(lock, [] { return !g_loads.empty() || !g_dumps.empty(); });
                    if (!g_loads.empty())
                    {
                        load = g_loads.front();
                        g_loads.pop_front();
                        path = g_files[load];
                        haveLoad = true;
                    }
                    else
                    {
                        dump = std::move(g_dumps.front());
                        g_dumps.pop_front();
                    }
                }
                if (haveLoad)
                {
                    size_t bytes = 0;
                    const auto start = std::chrono::steady_clock::now();
                    std::shared_ptr<const Image> image = loadImage(path, load, bytes);
                    g_statLoadNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - start).count()),
                                           std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(g_mutex);
                    CacheEntry &e = g_cache[load];
                    e.queued = false;
                    if (!image)
                    {
                        e.failed = true;
                        std::fprintf(stderr, "[tex-pack] failed to load %s\n", path.string().c_str());
                        continue;
                    }
                    e.image = std::move(image);
                    e.bytes = bytes;
                    g_cacheBytes += bytes;
                    g_statLoads.fetch_add(1, std::memory_order_relaxed);
                    evictLocked();
                    g_loadGen.fetch_add(1, std::memory_order_release);
                    continue;
                }
                std::error_code ec;
                if (fs::exists(dump.path, ec))
                    continue;
                ::Image out{};
                out.data = dump.rgba.data();
                out.width = dump.w;
                out.height = dump.h;
                out.mipmaps = 1;
                out.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
                if (::ExportImage(out, dump.path.string().c_str()))
                    g_statDumps.fetch_add(1, std::memory_order_relaxed);
            }
        }

        void startWorkerLocked()
        {
            if (g_workerStarted)
                return;
            g_workerStarted = true;
            std::thread(workerMain).detach();
        }

        void logStats()
        {
            if (!config().debug)
                return;
            static int64_t s_next = 0;
            const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now < s_next)
                return;
            s_next = now + 10;
            size_t cached = 0, files = 0;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                cached = g_cacheBytes;
                files = g_files.size();
            }
            std::fprintf(stderr,
                         "[tex-pack] hashes=%llu hash_ms=%.1f slow_path=%llu fuzzy=%llu/%llu fuzzy_ms=%.1f hit_prims=%llu loads=%llu "
                         "load_ms=%.0f dumps=%llu cache_mb=%.1f pack_files=%zu\n",
                         static_cast<unsigned long long>(g_statHashes.load()), static_cast<double>(g_statHashNs.load()) / 1e6,
                         static_cast<unsigned long long>(g_statSlowPath.load()), static_cast<unsigned long long>(g_statFuzzy.load()),
                         static_cast<unsigned long long>(g_statFuzzyTries.load()), static_cast<double>(g_statFuzzyNs.load()) / 1e6,
                         static_cast<unsigned long long>(g_statHits.load()), static_cast<unsigned long long>(g_statLoads.load()),
                         static_cast<double>(g_statLoadNs.load()) / 1e6, static_cast<unsigned long long>(g_statDumps.load()),
                         static_cast<double>(cached) / (1024.0 * 1024.0), files);
        }

        // Texels as PCSX2 dumps them (psm_t::rtx: CLUT and TEXA applied, PS2 alpha).
        void queueDump(const Name &name, const Texture &t, const GSDrawState &state, const uint8_t *vram,
                       const uint32_t (&palette)[256])
        {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                if (!g_dumped.insert(name).second)
                    return;
            }
            const auto &tex0 = state.context.tex0;
            DumpJob job;
            job.path = config().dump / fileName(name);
            job.w = t.tw;
            job.h = t.th;
            job.rgba.resize(static_cast<size_t>(t.tw) * t.th);
            for (int y = 0; y < t.th; ++y)
                for (int x = 0; x < t.tw; ++x)
                {
                    const uint32_t raw = readTexel(t, tex0, vram, t.rect.l + x, t.rect.t + y);
                    uint32_t c = raw;
                    switch (t.p.fmt)
                    {
                    case Fmt::Pal:
                        c = palette[raw & (t.p.pal - 1u)];
                        break;
                    case Fmt::C24:
                        c = expand24(raw, state.texa);
                        break;
                    case Fmt::C16:
                        c = expand16(raw & 0xFFFFu, state.texa);
                        break;
                    default:
                        break;
                    }
                    job.rgba[static_cast<size_t>(y) * t.tw + x] = c;
                }
            std::lock_guard<std::mutex> lock(g_mutex);
            g_dumps.push_back(std::move(job));
            startWorkerLocked();
            g_cv.notify_one();
        }

        // Box thumbnail of the texture as the pack's thumbnails were made (RGB after TEXA).
        bool thumbnail(const Texture &t, const GSDrawState &state, const uint8_t *vram, Thumb &out)
        {
            if (t.p.fmt == Fmt::Pal || t.tw < kThumb || t.th < kThumb)
                return false;
            const auto &tex0 = state.context.tex0;
            const int bw = t.tw / kThumb, bh = t.th / kThumb;
            std::array<uint32_t, kThumb * kThumb * 3> sum{};
            for (int y = 0; y < bh * kThumb; ++y)
                for (int x = 0; x < bw * kThumb; ++x)
                {
                    const uint32_t raw = readTexel(t, tex0, vram, t.rect.l + x, t.rect.t + y);
                    const uint32_t c = t.p.fmt == Fmt::C16 ? expand16(raw & 0xFFFFu, state.texa)
                                       : t.p.fmt == Fmt::C24 ? expand24(raw, state.texa) : raw;
                    uint32_t *cell = &sum[static_cast<size_t>((y / bh) * kThumb + x / bw) * 3];
                    cell[0] += c & 0xFFu;
                    cell[1] += (c >> 8) & 0xFFu;
                    cell[2] += (c >> 16) & 0xFFu;
                }
            const uint32_t n = static_cast<uint32_t>(bw * bh);
            for (size_t i = 0; i < sum.size(); ++i)
                out[i] = static_cast<uint8_t>((sum[i] + n / 2) / n);
            return true;
        }

        // The pack texture with the same format, size and TEXA whose thumbnail is clearly the
        // closest (mean difference <= 8 of 255 and under half the runner-up's).
        bool fuzzyMatch(const Name &name, const Thumb &thumb, Name &out)
        {
            const auto it = g_fuzzy.find(name.bits);
            if (it == g_fuzzy.end())
                return false;
            uint32_t best = UINT32_MAX, second = UINT32_MAX;
            const FuzzyEntry *found = nullptr;
            for (const FuzzyEntry &e : it->second)
            {
                uint32_t d = 0;
                for (size_t i = 0; i < thumb.size(); ++i)
                    d += static_cast<uint32_t>(std::abs(static_cast<int>(thumb[i]) - static_cast<int>(e.thumb[i])));
                if (d < best)
                {
                    second = best;
                    best = d;
                    found = &e;
                }
                else if (d < second)
                    second = d;
            }
            const uint32_t limit = 8u * static_cast<uint32_t>(thumb.size());
            if (!found || best > limit || (second != UINT32_MAX && second < 2u * best + static_cast<uint32_t>(thumb.size())))
                return false;
            out = found->name;
            if (config().debug)
                std::fprintf(stderr, "[tex-pack] fuzzy %s -> %s (diff %.2f, next %.2f)\n", fileName(name).c_str(),
                             fileName(found->name).c_str(), static_cast<double>(best) / thumb.size(),
                             second == UINT32_MAX ? -1.0 : static_cast<double>(second) / thumb.size());
            return true;
        }

        bool packHas(const Name &name)
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            return g_files.find(name) != g_files.end();
        }

        std::shared_ptr<const Image> findReplacement(const Name &name)
        {
            if (!g_indexed.load(std::memory_order_acquire))
                return nullptr;
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_files.find(name) == g_files.end())
                return nullptr;
            CacheEntry &e = g_cache[name];
            e.lastUse = ++g_useClock;
            if (e.image)
                return e.image;
            if (!e.queued && !e.failed)
            {
                e.queued = true;
                g_loads.push_back(name);
                g_cv.notify_one();
            }
            return nullptr;
        }

        // ---- per-backend state (submit thread, except inFlight) -------------------------

        struct TexKey
        {
            uint64_t tex = 0, clamp = 0;
            bool operator==(const TexKey &o) const { return tex == o.tex && clamp == o.clamp; }
        };

        struct TexKeyHash
        {
            size_t operator()(const TexKey &k) const { return static_cast<size_t>(k.tex * 0x9E3779B97F4A7C15ull ^ k.clamp); }
        };

        struct TexEntry
        {
            uint64_t hash = 0, blocks = 0; // PCSX2 hash; hashBlocks() when it was taken
            uint32_t firstPage = 0, lastPage = 0;
            uint64_t versionSum = 0, clock = 0;
            bool valid = false;  // describe() accepted it
            bool target = false; // drawn into since uploaded: PCSX2 samples the render target, never a replacement
        };

        struct NameEntry
        {
            std::weak_ptr<const Image> image; // the cache owns it; eviction frees it once queued draws are done
            uint64_t loadGen = 0;
            Name target;                     // pack name to load: the same, or a thumbnail match
            bool resolved = false;           // target settled (exact name, match, or none)
        };

        struct ClutKey
        {
            uint64_t gen = 0, sel = 0;
            bool operator==(const ClutKey &o) const { return gen == o.gen && sel == o.sel; }
        };

        struct State
        {
            const void *owner = nullptr;
            // Page versions: bumped when something writes a page a cached hash read.
            std::array<uint32_t, kPages> version{};
            std::array<uint64_t, kPages / 64> watched{};
            std::array<uint64_t, kPages / 64> rendered{}; // drawn into since their last upload
            uint64_t clock = 0, watchGen = 0, uploadGen = 0;

            std::unordered_map<TexKey, TexEntry, TexKeyHash> tex;
            std::vector<std::pair<ClutKey, uint64_t>> cluts;
            uint64_t clutGen = 0;
            std::unordered_map<Name, NameEntry, NameHash> names;

            // Last primitive, for runs of primitives with one texture.
            bool lastValid = false;
            TexKey lastTex;
            ClutKey lastClut;
            uint64_t lastClock = 0, lastLoadGen = 0;
            std::shared_ptr<const Image> lastImage;

            // Last FRAME/ZBUF checked against watched pages.
            uint64_t lastTargets = ~0ull, lastTargetsWatchGen = ~0ull, lastTargetsUploadGen = ~0ull;

            // Queued primitives' replacements, index for index with the band queue.
            // The *Refs lists hold each image once per batch so eviction cannot free it under a band.
            const Image *submit = nullptr; // the primitive being submitted (owned by lastImage)
            std::vector<const Image *> pending, inFlight;
            std::vector<std::shared_ptr<const Image>> pendingRefs, inFlightRefs;
            const Image *lastPendingRef = nullptr;
            uint64_t hitPrims = 0;
        };

        std::mutex g_statesMutex;
        std::deque<State> g_states;

        State &stateFor(const void *owner)
        {
            thread_local State *t_last = nullptr;
            if (t_last && t_last->owner == owner)
                return *t_last;
            std::lock_guard<std::mutex> lock(g_statesMutex);
            for (State &s : g_states)
                if (s.owner == owner)
                    return *(t_last = &s);
            g_states.emplace_back();
            g_states.back().owner = owner;
            return *(t_last = &g_states.back());
        }

        uint64_t versionSum(const State &s, uint32_t first, uint32_t last)
        {
            uint64_t sum = 0;
            for (uint32_t p = first; p <= last && p < kPages; ++p)
                sum += s.version[p];
            return sum;
        }

        void watch(State &s, uint32_t first, uint32_t last)
        {
            for (uint32_t p = first; p <= last && p < kPages; ++p)
                s.watched[p >> 6] |= 1ull << (p & 63u);
            ++s.watchGen;
        }

        bool anyRendered(const State &s, uint32_t first, uint32_t last)
        {
            for (uint32_t p = first; p <= last && p < kPages; ++p)
                if (s.rendered[p >> 6] & (1ull << (p & 63u)))
                    return true;
            return false;
        }

        void markRendered(State &s, uint32_t first, uint32_t endPage, bool rendered)
        {
            for (uint32_t p = first; p < std::min(endPage, kPages); ++p)
            {
                if (rendered)
                    s.rendered[p >> 6] |= 1ull << (p & 63u);
                else
                    s.rendered[p >> 6] &= ~(1ull << (p & 63u));
            }
        }

        void bumpPages(State &s, uint32_t first, uint32_t endPage)
        {
            endPage = std::min(endPage, kPages);
            for (uint32_t p = first; p < endPage;)
            {
                const uint32_t word = p >> 6;
                if (s.watched[word] == 0u)
                {
                    p = (word + 1u) << 6;
                    continue;
                }
                const uint64_t bit = 1ull << (p & 63u);
                if (s.watched[word] & bit)
                {
                    s.watched[word] &= ~bit;
                    ++s.version[p];
                    ++s.clock;
                }
                ++p;
            }
        }

        thread_local const Image *t_submitImage = nullptr;
        thread_local const State *t_bandState = nullptr;
        thread_local const Image *t_bandImage = nullptr;
    }

    void NoteVramWrite(const void *owner, uint32_t begin, uint32_t end)
    {
        State &s = stateFor(owner);
        uint32_t first = 0u, endPage = kPages;
        if (end != UINT32_MAX && end <= kPages * kPageBytes && begin < end)
        {
            first = begin / kPageBytes;
            endPage = (end + kPageBytes - 1u) / kPageBytes;
        }
        bumpPages(s, first, endPage);
        markRendered(s, first, endPage, false);
        ++s.uploadGen;
    }

    void NoteTargets(const void *owner, const GSContext &ctx)
    {
        State &s = stateFor(owner);
        const uint32_t fbw = ctx.frame.fbw >= 1u ? ctx.frame.fbw : 10u;
        const uint32_t rows = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
        const uint64_t key = static_cast<uint64_t>(ctx.frame.fbp & 0x1FFu) | (static_cast<uint64_t>(fbw & 0x3Fu) << 9) |
                             (static_cast<uint64_t>(ctx.frame.psm & 0x3Fu) << 15) | (static_cast<uint64_t>(rows & 0xFFFu) << 21) |
                             (static_cast<uint64_t>(ctx.zbuf.zbp & 0x1FFu) << 33) | (static_cast<uint64_t>(ctx.zbuf.psm & 0x3Fu) << 42) |
                             (static_cast<uint64_t>(ctx.zbuf.zmask) << 48);
        if (key == s.lastTargets && s.watchGen == s.lastTargetsWatchGen && s.uploadGen == s.lastTargetsUploadGen)
            return;
        s.lastTargets = key;
        const auto surface = [&](uint32_t page, uint32_t psm)
        {
            const bool sixteen = psm == GS_PSM_CT16 || psm == GS_PSM_CT16S || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S;
            const uint32_t pageRows = sixteen ? 64u : 32u;
            const uint32_t pages = (rows + pageRows - 1u) / pageRows * fbw;
            bumpPages(s, page, page + pages);
            markRendered(s, page, page + pages, true);
        };
        surface(ctx.frame.fbp, ctx.frame.psm);
        if (!ctx.zbuf.zmask)
            surface(ctx.zbuf.zbp, ctx.zbuf.psm);
        s.lastTargetsWatchGen = s.watchGen;
        s.lastTargetsUploadGen = s.uploadGen;
    }

    void NoteClutLoad(const void *owner)
    {
        ++stateFor(owner).clutGen;
    }

    void OnSubmit(const void *owner, const GSPrimitiveBatch &batch, const uint8_t *vram, const uint16_t *clut)
    {
        State &s = stateFor(owner);
        const GSDrawState &state = batch.state;
        const auto &tex0 = state.context.tex0;
        const uint64_t clamp = state.context.clamp;
        s.submit = nullptr;
        t_submitImage = nullptr;
        // Region repeat remaps texel addresses bitwise; leave it to the GS sampler.
        if (state.prim.tme && (clamp & 3u) != 3u && ((clamp >> 2) & 3u) != 3u)
        {
            const bool palette = isPalettePsm(tex0.psm);
            TexKey tk;
            tk.tex = static_cast<uint64_t>(tex0.tbp0 & 0x3FFFu) | (static_cast<uint64_t>(tex0.tbw & 0x3Fu) << 14) |
                     (static_cast<uint64_t>(tex0.psm & 0x3Fu) << 20) | (static_cast<uint64_t>(tex0.tw & 0xFu) << 26) |
                     (static_cast<uint64_t>(tex0.th & 0xFu) << 30);
            if (!palette && tex0.psm != GS_PSM_CT32)
                tk.tex |= (static_cast<uint64_t>(state.texa.ta0) << 34) | (static_cast<uint64_t>(state.texa.aem) << 42) |
                          (static_cast<uint64_t>(state.texa.ta1) << 43);
            tk.clamp = clamp & 0xFFFFFFFFFFFull;
            ClutKey ck;
            if (palette)
            {
                ck.gen = s.clutGen;
                ck.sel = static_cast<uint64_t>(tex0.cpsm & 0x3Fu) | (static_cast<uint64_t>(tex0.csa & 0x1Fu) << 6) |
                         (static_cast<uint64_t>(tex0.psm == GS_PSM_T8 || tex0.psm == GS_PSM_T8H) << 11) |
                         (static_cast<uint64_t>(state.texa.ta0) << 12) | (static_cast<uint64_t>(state.texa.aem) << 20) |
                         (static_cast<uint64_t>(state.texa.ta1) << 21) | 1ull << 63;
            }
            const uint64_t loadGen = g_loadGen.load(std::memory_order_acquire);
            if (!(s.lastValid && s.lastTex == tk && s.lastClut == ck && s.lastClock == s.clock && s.lastLoadGen == loadGen))
            {
                if (config().debug)
                    g_statSlowPath.fetch_add(1, std::memory_order_relaxed);
                Texture t;
                std::shared_ptr<const Image> image;
                if (describe(state, t))
                {
                    Name name;
                    name.bits = nameBits(state, t.p);
                    if (t.region.hasX() || t.region.hasY())
                    {
                        name.regionW = static_cast<uint32_t>(t.region.maxX - t.region.minX);
                        name.regionH = static_cast<uint32_t>(t.region.maxY - t.region.minY);
                    }

                    TexEntry &e = s.tex[tk];
                    if (!e.valid || (e.clock != s.clock && versionSum(s, e.firstPage, e.lastPage) != e.versionSum))
                    {
                        const bool wasHashed = e.valid && !e.target;
                        pageRange(t, tex0, e.firstPage, e.lastPage);
                        e.target = anyRendered(s, e.firstPage, e.lastPage);
                        if (!e.target)
                        {
                            const auto start = config().debug ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                            // The game re-uploads the same texels every frame: only the slow
                            // expanded hash is skipped when the raw blocks did not change.
                            const uint64_t blocks = hashBlocks(t, tex0, vram);
                            if (!wasHashed || blocks != e.blocks)
                            {
                                e.hash = hashesBlocks(t) ? blocks : hashTexture(t, state, vram);
                                e.blocks = blocks;
                                g_statHashes.fetch_add(1, std::memory_order_relaxed);
                            }
                            if (config().debug)
                                g_statHashNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                           std::chrono::steady_clock::now() - start).count()),
                                                       std::memory_order_relaxed);
                        }
                        e.valid = true;
                        watch(s, e.firstPage, e.lastPage);
                        e.versionSum = versionSum(s, e.firstPage, e.lastPage);
                    }
                    e.clock = s.clock;
                    name.tex0Hash = e.hash;

                    uint32_t paletteRgba[256];
                    uint32_t paletteCount = 0;
                    bool paletteOk = !e.target;
                    if (palette && paletteOk)
                    {
                        paletteOk = readClut32(tex0, state.texa, clut, paletteRgba, paletteCount);
                        bool found = false;
                        for (const auto &c : s.cluts)
                            if (c.first == ck)
                            {
                                name.clutHash = c.second;
                                found = true;
                                break;
                            }
                        if (!found && paletteOk)
                        {
                            name.clutHash = XXH3_64bits(paletteRgba, paletteCount * sizeof(uint32_t));
                            if (s.cluts.size() >= 256u)
                                s.cluts.clear();
                            s.cluts.push_back({ck, name.clutHash});
                        }
                    }

                    if (paletteOk)
                    {
                        auto it = s.names.find(name);
                        if (it == s.names.end())
                        {
                            if (!config().dump.empty())
                                queueDump(name, t, state, vram, paletteRgba);
                            it = s.names.emplace(name, NameEntry{}).first;
                            it->second.loadGen = ~0ull;
                        }
                        NameEntry &ne = it->second;
                        image = ne.image.lock();
                        if (!config().packs.empty() && !image && ne.loadGen != loadGen)
                        {
                            {
                                std::lock_guard<std::mutex> lock(g_mutex);
                                startWorkerLocked();
                            }
                            if (!ne.resolved && g_indexed.load(std::memory_order_acquire))
                            {
                                ne.target = name;
                                ne.resolved = true;
                                Thumb thumb;
                                Name match;
                                const auto start = std::chrono::steady_clock::now();
                                if (!packHas(name) && g_fuzzy.count(name.bits) != 0 && thumbnail(t, state, vram, thumb))
                                {
                                    g_statFuzzyTries.fetch_add(1, std::memory_order_relaxed);
                                    if (fuzzyMatch(name, thumb, match))
                                    {
                                        ne.target = match;
                                        g_statFuzzy.fetch_add(1, std::memory_order_relaxed);
                                    }
                                    g_statFuzzyNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                std::chrono::steady_clock::now() - start).count()),
                                                            std::memory_order_relaxed);
                                }
                            }
                            if (ne.resolved)
                                image = findReplacement(ne.target);
                            ne.image = image;
                            ne.loadGen = loadGen;
                        }
                    }
                    logStats();
                }
                s.lastValid = true;
                s.lastTex = tk;
                s.lastClut = ck;
                s.lastClock = s.clock;
                s.lastLoadGen = loadGen;
                s.lastImage = std::move(image);
            }
            s.submit = s.lastImage.get();
            s.hitPrims += s.submit != nullptr;
            t_submitImage = s.submit;
        }
        // This primitive draws after sampling: a later one sampling its pages re-hashes.
        NoteTargets(owner, state.context);
    }

    void OnQueued(const void *owner)
    {
        State &s = stateFor(owner);
        s.pending.push_back(s.submit);
        if (s.submit && s.submit != s.lastPendingRef)
        {
            s.pendingRefs.push_back(s.lastImage);
            s.lastPendingRef = s.submit;
        }
    }

    void OnBandFlush(const void *owner)
    {
        State &s = stateFor(owner);
        s.inFlight.swap(s.pending);
        s.pending.clear();
        s.inFlightRefs.swap(s.pendingRefs);
        s.pendingRefs.clear();
        s.lastPendingRef = nullptr;
        g_statHits.fetch_add(s.hitPrims, std::memory_order_relaxed);
        s.hitPrims = 0;
    }

    void BeginBand(const void *owner)
    {
        t_bandState = &stateFor(owner);
        t_bandImage = nullptr;
    }

    void SetBandDraw(uint32_t index)
    {
        const State *s = t_bandState;
        t_bandImage = (s && index < s->inFlight.size()) ? s->inFlight[index] : nullptr;
    }

    void EndBand()
    {
        t_bandState = nullptr;
        t_bandImage = nullptr;
    }

    const Image *Current()
    {
        return t_bandState ? t_bandImage : t_submitImage;
    }

    uint32_t PickLevel(const Image &img, const GSPrimitiveBatch &batch)
    {
        const GSDrawState &state = batch.state;
        const float texW = static_cast<float>(state.textureWidth), texH = static_cast<float>(state.textureHeight);
        const uint32_t baseW = img.regionW ? img.regionW : state.textureWidth;
        const uint32_t baseH = img.regionH ? img.regionH : state.textureHeight;
        uint32_t cap = 0;
        while (cap + 1u < img.levelCount && img.levels[cap + 1u].w >= baseW && img.levels[cap + 1u].h >= baseH)
            ++cap;
        // ATV2_TEXTURE_PACK_SCALE: finest level used, as a multiple of the GS texture's size.
        uint32_t finest = 0;
        while (finest < cap && img.levels[finest].w > baseW * config().maxScale)
            ++finest;
        if (cap == finest || batch.vertexCount < 2u)
            return finest;

        float u[3], v[3];
        for (uint32_t i = 0; i < batch.vertexCount && i < 3u; ++i)
        {
            const GSVertex &vx = batch.vertices[i];
            if (state.prim.fst)
            {
                u[i] = static_cast<float>(vx.u) / 16.0f;
                v[i] = static_cast<float>(vx.v) / 16.0f;
            }
            else
            {
                const float q = vx.q > 1e-5f ? vx.q : 1.0f;
                u[i] = vx.s / q * texW;
                v[i] = vx.t / q * texH;
            }
        }
        // GS texels per screen pixel.
        float density;
        if (batch.vertexCount == 2u)
        {
            const float dx = std::fabs(batch.vertices[1].x - batch.vertices[0].x);
            const float dy = std::fabs(batch.vertices[1].y - batch.vertices[0].y);
            density = std::max(dx > 0.0f ? std::fabs(u[1] - u[0]) / dx : 0.0f, dy > 0.0f ? std::fabs(v[1] - v[0]) / dy : 0.0f);
        }
        else
        {
            const GSVertex &a = batch.vertices[0], &b = batch.vertices[1], &c = batch.vertices[2];
            const float screen = std::fabs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
            const float texels = std::fabs((u[1] - u[0]) * (v[2] - v[0]) - (v[1] - v[0]) * (u[2] - u[0]));
            if (screen < 1e-6f)
                return cap;
            density = std::sqrt(texels / screen);
        }
        const float scale = static_cast<float>(img.levels[0].w) / static_cast<float>(baseW);
        const float lod = std::log2(std::max(density * scale, 1e-6f));
        return static_cast<uint32_t>(std::clamp(static_cast<int>(std::floor(lod + 0.5f)), static_cast<int>(finest), static_cast<int>(cap)));
    }

    void Prepare(const Image &img, uint32_t level, const GSDrawState &state, Sampler &out)
    {
        const Level &l = img.levels[std::min(level, img.levelCount - 1u)];
        out.texels = l.texels;
        out.pitch = l.w;
        out.linear = state.linearFilter;
        const uint64_t clamp = state.context.clamp;
        const auto setup = [](Axis &a, uint32_t mode, uint32_t gsSize, uint32_t regionSize, uint32_t minC, uint32_t maxC, uint32_t size)
        {
            a.size = static_cast<int>(size);
            a.pow2 = (size & (size - 1u)) == 0u;
            a.offset = 0.0f;
            a.lo = 0;
            a.hi = static_cast<int>(size) - 1;
            a.repeat = mode == 0u;
            if (mode == 2u && regionSize != 0u)
            {
                // The image holds texels [min, max] of the GS texture.
                a.scale = static_cast<float>(size) / static_cast<float>(regionSize);
                a.offset = -static_cast<float>(minC) * a.scale;
                return;
            }
            a.scale = static_cast<float>(size) / static_cast<float>(std::max(gsSize, 1u));
            if (mode == 2u)
            {
                a.lo = std::clamp(static_cast<int>(std::floor(static_cast<float>(minC) * a.scale)), 0, a.hi);
                a.hi = std::clamp(static_cast<int>(std::ceil(static_cast<float>(maxC + 1u) * a.scale)) - 1, a.lo, a.hi);
            }
        };
        setup(out.u, clamp & 3u, state.textureWidth, img.regionW, (clamp >> 4) & 0x3FFu, (clamp >> 14) & 0x3FFu, l.w);
        setup(out.v, (clamp >> 2) & 3u, state.textureHeight, img.regionH, (clamp >> 24) & 0x3FFu, (clamp >> 34) & 0x3FFu, l.h);
    }
}
