// Suzuki DZ400 in the ATV2 menus. The race archives QUAD\BIKESA\ATV06(_MT).RES are on the disc and
// the quad drives fine as gameplay ID 6, but the menus never offer it: the roster uiQuad.txt has no
// IngameID 6 entry and the disc has no UI\RESOURCE\QUAD\ATV06.RES menu model (the menu string table
// in LOADUI.RES still names logo_suzuki, multi_atv06 and SuzDZ400_*_ui_hires, so it was cut late).
// atv2BuildSuzukiDz400Files builds, from the retail files, the disc files that list it in ATV & Rider
// Select as the last roster entry; scus_97211_overrides.cpp serves them in place of the disc's (and
// picks its race opponents, which quad\ATVGroups.pb never classed).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "raylib.h" // CompressData / DecompressData (raw DEFLATE)

namespace
{
#include "Atv2SuzukiDz400Logo.inc"

    // Rainbow Studios .RES archive: u32 3, u32 tableBytes + 12, u32 832, u32 flags, u32 tableBytes,
    // u16 compressed block sizes ending in 0, u32 fileCount, u32 recordBytes, then per file
    // {u32 nameLength, name, u32 offset, u32 size}, then the blocks. Each block is a zlib stream that
    // inflates to 0x6000 bytes of one solid stream and is followed by one 0xFF byte; file offsets
    // index the inflated stream and every file starts on a block.
    constexpr uint32_t kResBlockBytes = 0x6000u;

    struct ResFile
    {
        std::string name;
        uint32_t offset = 0u;
        uint32_t size = 0u;
    };

    struct ResArchive
    {
        uint32_t field8 = 832u;
        uint32_t flags = 0u;
        std::vector<std::vector<uint8_t>> blocks;
        std::vector<ResFile> files;
    };

    bool sameName(const std::string &a, const std::string &b)
    {
        return a.size() == b.size() &&
               std::equal(a.begin(), a.end(), b.begin(), [](char x, char y)
                          { return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y)); });
    }

    bool parseRes(const std::vector<uint8_t> &data, ResArchive &res)
    {
        size_t pos = 0;
        auto u32 = [&](uint32_t &value)
        {
            if (pos + 4u > data.size())
                return false;
            std::memcpy(&value, data.data() + pos, 4u);
            pos += 4u;
            return true;
        };
        uint32_t version = 0, headerBytes = 0, tableBytes = 0, count = 0, recordBytes = 0;
        if (!u32(version) || version != 3u || !u32(headerBytes) || !u32(res.field8) || !u32(res.flags) ||
            !u32(tableBytes) || pos + tableBytes > data.size())
            return false;
        std::vector<uint16_t> sizes;
        for (uint32_t i = 0; i + 1u < tableBytes; i += 2u)
        {
            uint16_t size = 0;
            std::memcpy(&size, data.data() + pos + i, 2u);
            if (size == 0u)
                break;
            sizes.push_back(size);
        }
        pos += tableBytes;
        if (!u32(count) || !u32(recordBytes))
            return false;
        for (uint32_t i = 0; i < count; ++i)
        {
            ResFile file;
            uint32_t nameLength = 0;
            if (!u32(nameLength) || pos + nameLength > data.size())
                return false;
            file.name.assign(reinterpret_cast<const char *>(data.data() + pos), nameLength);
            pos += nameLength;
            if (!u32(file.offset) || !u32(file.size))
                return false;
            res.files.push_back(std::move(file));
        }
        for (uint16_t size : sizes)
        {
            if (pos + size + 1u > data.size() || data[pos + size] != 0xFFu)
                return false;
            res.blocks.emplace_back(data.begin() + pos, data.begin() + pos + size);
            pos += size + 1u;
        }
        return true;
    }

    std::vector<uint8_t> serializeRes(const ResArchive &res)
    {
        std::vector<uint8_t> out;
        auto put = [&](uint32_t value, int bytes)
        {
            for (int i = 0; i < bytes; ++i)
                out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        };
        const uint32_t tableBytes = static_cast<uint32_t>(res.blocks.size() + 1u) * 2u;
        put(3u, 4);
        put(tableBytes + 12u, 4);
        put(res.field8, 4);
        put(res.flags, 4);
        put(tableBytes, 4);
        for (const std::vector<uint8_t> &block : res.blocks)
            put(static_cast<uint32_t>(block.size()), 2);
        put(0u, 2);
        uint32_t recordBytes = 0;
        for (const ResFile &file : res.files)
            recordBytes += 12u + static_cast<uint32_t>(file.name.size());
        put(static_cast<uint32_t>(res.files.size()), 4);
        put(recordBytes, 4);
        for (const ResFile &file : res.files)
        {
            put(static_cast<uint32_t>(file.name.size()), 4);
            out.insert(out.end(), file.name.begin(), file.name.end());
            put(file.offset, 4);
            put(file.size, 4);
        }
        for (const std::vector<uint8_t> &block : res.blocks)
        {
            out.insert(out.end(), block.begin(), block.end());
            out.push_back(0xFFu);
        }
        return out;
    }

    uint32_t adler32(const uint8_t *data, size_t size)
    {
        uint32_t a = 1u, b = 0u;
        for (size_t i = 0; i < size; ++i)
        {
            a = (a + data[i]) % 65521u;
            b = (b + a) % 65521u;
        }
        return (b << 16) | a;
    }

    // zlib stream (2-byte header, raw DEFLATE, adler32) <-> raylib's raw DEFLATE.
    bool inflateZlib(const uint8_t *stream, size_t size, std::vector<uint8_t> &out)
    {
        if (size < 6u || (stream[0] & 0x0Fu) != 8u)
            return false;
        int length = 0;
        unsigned char *data = DecompressData(stream + 2, static_cast<int>(size - 6u), &length);
        if (data == nullptr)
            return false;
        out.insert(out.end(), data, data + length);
        MemFree(data);
        return length > 0;
    }

    std::vector<uint8_t> deflateBlock(const uint8_t *data)
    {
        int length = 0;
        unsigned char *raw = CompressData(data, static_cast<int>(kResBlockBytes), &length);
        std::vector<uint8_t> block{0x78u, 0x9Cu};
        block.insert(block.end(), raw, raw + length);
        MemFree(raw);
        const uint32_t sum = adler32(data, kResBlockBytes);
        for (int shift = 24; shift >= 0; shift -= 8)
            block.push_back(static_cast<uint8_t>(sum >> shift));
        return block;
    }

    const ResFile *findFile(const ResArchive &res, const std::string &name)
    {
        for (const ResFile &file : res.files)
        {
            if (sameName(file.name, name))
                return &file;
        }
        return nullptr;
    }

    bool extractFile(const ResArchive &res, const ResFile &file, std::vector<uint8_t> &out)
    {
        std::vector<uint8_t> stream;
        const uint32_t first = file.offset / kResBlockBytes;
        const uint32_t last = (file.offset + std::max(file.size, 1u) - 1u) / kResBlockBytes;
        for (uint32_t i = first; i <= last; ++i)
        {
            const size_t before = stream.size();
            if (i >= res.blocks.size() || !inflateZlib(res.blocks[i].data(), res.blocks[i].size(), stream) ||
                stream.size() - before != kResBlockBytes)
                return false;
        }
        const uint32_t skip = file.offset - first * kResBlockBytes;
        out.assign(stream.begin() + skip, stream.begin() + skip + file.size);
        return true;
    }

    // Blocks [firstBlock, firstBlock + blockCount) hold data, zero padded.
    void writeBlocks(ResArchive &res, size_t firstBlock, size_t blockCount, const std::vector<uint8_t> &data)
    {
        std::vector<uint8_t> chunk(kResBlockBytes);
        for (size_t i = 0; i < blockCount; ++i)
        {
            std::fill(chunk.begin(), chunk.end(), 0u);
            const size_t begin = i * kResBlockBytes;
            if (begin < data.size())
                std::memcpy(chunk.data(), data.data() + begin, std::min<size_t>(kResBlockBytes, data.size() - begin));
            if (firstBlock + i < res.blocks.size())
                res.blocks[firstBlock + i] = deflateBlock(chunk.data());
            else
                res.blocks.push_back(deflateBlock(chunk.data()));
        }
    }

    void appendFile(ResArchive &res, const std::string &name, const std::vector<uint8_t> &data)
    {
        const size_t first = res.blocks.size();
        writeBlocks(res, first, std::max<size_t>(1u, (data.size() + kResBlockBytes - 1u) / kResBlockBytes), data);
        res.files.push_back({name, static_cast<uint32_t>(first * kResBlockBytes), static_cast<uint32_t>(data.size())});
    }

    // Copies a file's compressed blocks from another archive.
    void appendFileBlocks(ResArchive &res, const std::string &name, const ResArchive &src, const ResFile &file)
    {
        const uint32_t first = file.offset / kResBlockBytes;
        const uint32_t last = (file.offset + std::max(file.size, 1u) - 1u) / kResBlockBytes;
        const uint32_t offset = static_cast<uint32_t>(res.blocks.size()) * kResBlockBytes + file.offset % kResBlockBytes;
        for (uint32_t i = first; i <= last && i < src.blocks.size(); ++i)
            res.blocks.push_back(src.blocks[i]);
        res.files.push_back({name, offset, file.size});
    }

    // Rewrites a file in place; it may grow up to the next file's first block.
    bool replaceFile(ResArchive &res, const std::string &name, const std::vector<uint8_t> &data)
    {
        ResFile *file = const_cast<ResFile *>(findFile(res, name));
        if (file == nullptr || file->offset % kResBlockBytes != 0u)
            return false;
        uint32_t end = static_cast<uint32_t>(res.blocks.size()) * kResBlockBytes;
        for (const ResFile &other : res.files)
        {
            if (other.offset > file->offset)
                end = std::min(end, other.offset - other.offset % kResBlockBytes);
        }
        if (data.size() > end - file->offset)
            return false;
        writeBlocks(res, file->offset / kResBlockBytes, (end - file->offset) / kResBlockBytes, data);
        file->size = static_cast<uint32_t>(data.size());
        return true;
    }

    // uiQuad.txt: one more [General] NumQuads and the DZ400 as the last [QuadN], so the roster
    // indices of the retail quads (profiles store them) stay put. Stats follow ATV06S01.phy: more power
    // than the KFX400 (peak 52 HP vs 47) and less tire friction (1.2 vs 1.45).
    bool addRosterEntry(std::vector<uint8_t> &file)
    {
        std::string text(file.begin(), file.end());
        for (size_t at = text.find("IngameID"); at != std::string::npos; at = text.find("IngameID", at + 8u))
        {
            const size_t value = text.find_first_not_of(" \t=", at + 8u);
            if (value != std::string::npos && text.compare(value, 2, "6\r") == 0)
                return false; // already listed
        }
        const size_t key = text.find("NumQuads");
        const size_t value = key == std::string::npos ? key : text.find_first_of("0123456789", key);
        if (value == std::string::npos)
            return false;
        const size_t valueEnd = text.find_first_not_of("0123456789", value);
        const int count = std::atoi(text.substr(value, valueEnd - value).c_str());
        if (count <= 0)
            return false;
        text.replace(value, valueEnd - value, std::to_string(count + 1));
        if (text.size() < 2u || text.compare(text.size() - 2u, 2u, "\r\n") != 0)
            text += "\r\n";
        text += "\r\n;-----------------------------------------------------------\r\n"
                "[Quad" + std::to_string(count + 1) + "]\r\n"
                "\r\n"
                "SLT = atv06.slt\r\n"
                "MpSLT = atv06.slt\r\n"
                "Name = \"Suzuki DZ400\"\r\n"
                "Image = logo_suzuki\r\n"
                "Access = 1\r\n"
                "IngameID = 6\r\n"
                "Engine_Range = 0.7\r\n"
                "Traction_Range = 0.45\r\n"
                "Suspension_Range = 0.5\r\n"
                "Points = 1000\r\n";
        file.assign(text.begin(), text.end());
        return true;
    }

    // Material (.psm) for another texture: the retail one with its TextureMap name swapped.
    bool renameMaterial(std::vector<uint8_t> &psm, const std::string &from, const std::string &to)
    {
        std::string text(psm.begin(), psm.end());
        const size_t at = text.find("= " + from);
        if (at == std::string::npos)
            return false;
        text.replace(at + 2u, from.size(), to);
        psm.assign(text.begin(), text.end());
        return true;
    }

    // multi_atv06 (multiplayer ATV picture): multi_atv07 shows the KFX400, the same quad as the
    // DZ400 under Kawasaki green; repaint the green Suzuki yellow. .pst: u32 1, u32 1, u32 mips,
    // then 256 RGBA CLUT entries.
    void paintKawasakiGreenSuzukiYellow(std::vector<uint8_t> &pst)
    {
        for (size_t entry = 12u; entry + 4u <= 12u + 1024u && entry + 4u <= pst.size(); entry += 4u)
        {
            const float r = pst[entry] / 255.0f, g = pst[entry + 1] / 255.0f, b = pst[entry + 2] / 255.0f;
            const float maxc = std::max({r, g, b}), minc = std::min({r, g, b});
            const float delta = maxc - minc;
            if (maxc <= 0.12f || delta / maxc <= 0.30f || maxc != g)
                continue;
            float hue = 60.0f * ((b - r) / delta + 2.0f); // green dominant
            if (hue < 61.0f || hue > 180.0f)
                continue;
            // HSV(48 degrees, s, v'): R = v', G = v' (1 - s * 0.2), B = v' (1 - s)
            const float sat = std::min(1.0f, delta / maxc * 1.05f);
            const float v = std::min(1.0f, maxc * 1.12f + 0.06f);
            const float rgb[3] = {v, v * (1.0f - sat * 0.2f), v * (1.0f - sat)};
            for (int c = 0; c < 3; ++c)
                pst[entry + c] = static_cast<uint8_t>(std::lround(rgb[c] * 255.0f));
        }
    }

    bool rebuildWithRoster(const std::vector<uint8_t> &disc, ResArchive &res)
    {
        if (!parseRes(disc, res))
            return false;
        const ResFile *file = findFile(res, "ui\\frontend\\uiQuad.txt");
        std::vector<uint8_t> roster;
        return file != nullptr && extractFile(res, *file, roster) && addRosterEntry(roster) &&
               replaceFile(res, file->name, roster);
    }
}

// readDiscFile(path, data) reads a retail file named as in the game's file table, e.g.
// "\\UI\\RESOURCE\\QUADINFO.RES;1". Fills out with (disc path, contents) for each file to serve.
bool atv2BuildSuzukiDz400Files(const std::function<bool(const std::string &, std::vector<uint8_t> &)> &readDiscFile,
                               std::vector<std::pair<std::string, std::vector<uint8_t>>> &out)
{
    auto fail = [](const char *what)
    {
        std::fprintf(stderr, "[ATV2:DZ400] %s; the Suzuki DZ400 stays out of the menus\n", what);
        return false;
    };
    std::vector<uint8_t> disc;

    // Roster: QUADINFO.RES (+ logo and multiplayer picture) and the two pages that carry their own copy.
    ResArchive quadInfo;
    if (!readDiscFile("\\UI\\RESOURCE\\QUADINFO.RES;1", disc) || !rebuildWithRoster(disc, quadInfo))
        return fail("QUADINFO.RES roster not patched");
    std::vector<uint8_t> logoPsm, logoPst, multiPsm, multiPst;
    const ResFile *kfxLogo = findFile(quadInfo, "ui\\frontend\\textures\\logo_kfx700.psm");
    const ResFile *kfxMultiPsm = findFile(quadInfo, "ui\\frontend\\textures\\multi_atv07.psm");
    const ResFile *kfxMultiPst = findFile(quadInfo, "ui\\frontend\\textures\\multi_atv07.pst");
    if (kfxLogo == nullptr || kfxMultiPsm == nullptr || kfxMultiPst == nullptr ||
        !extractFile(quadInfo, *kfxLogo, logoPsm) || !renameMaterial(logoPsm, "logo_kfx700", "logo_suzuki") ||
        !extractFile(quadInfo, *kfxMultiPsm, multiPsm) || !renameMaterial(multiPsm, "multi_atv07", "multi_atv06") ||
        !extractFile(quadInfo, *kfxMultiPst, multiPst) ||
        !inflateZlib(kSuzukiLogoPstZlib, sizeof(kSuzukiLogoPstZlib), logoPst))
        return fail("QUADINFO.RES KFX700 textures not found");
    paintKawasakiGreenSuzukiYellow(multiPst);
    appendFile(quadInfo, "ui\\frontend\\textures\\logo_suzuki.psm", logoPsm);
    appendFile(quadInfo, "ui\\frontend\\textures\\logo_suzuki.pst", logoPst);
    appendFile(quadInfo, "ui\\frontend\\textures\\multi_atv06.psm", multiPsm);
    appendFile(quadInfo, "ui\\frontend\\textures\\multi_atv06.pst", multiPst);
    out.emplace_back("\\UI\\RESOURCE\\QUADINFO.RES;1", serializeRes(quadInfo));

    for (const char *page : {"\\UI\\RESOURCE\\PAGEA\\PAGE07.RES;1", "\\UI\\RESOURCE\\PAGEA\\PAGE20.RES;1"})
    {
        ResArchive res;
        if (!readDiscFile(page, disc) || !rebuildWithRoster(disc, res))
            return fail("page roster not patched");
        out.emplace_back(page, serializeRes(res));
    }

    // Menu model: the race model and its textures under the menu's paths.
    ResArchive model, textures, menu;
    std::vector<uint8_t> textureDisc;
    const ResFile *psg = nullptr;
    if (!readDiscFile("\\QUAD\\BIKESA\\ATV06.RES;1", disc) || !parseRes(disc, model) ||
        (psg = findFile(model, "quad\\atv06.psg")) == nullptr ||
        !readDiscFile("\\QUAD\\BIKESA\\ATV06_MT.RES;1", textureDisc) || !parseRes(textureDisc, textures))
        return fail("QUAD\\BIKESA\\ATV06 archives not readable");
    menu.field8 = model.field8;
    appendFileBlocks(menu, "ui\\frontend\\geometry\\atv06.psg", model, *psg);
    for (const ResFile &file : textures.files)
    {
        const std::string leaf = file.name.substr(file.name.find_last_of('\\') + 1u);
        if (leaf.rfind("arrow.", 0) != 0) // the race's wrong-way arrow
            appendFileBlocks(menu, "ui\\frontend\\textures\\" + leaf, textures, file);
    }
    out.emplace_back("\\UI\\RESOURCE\\QUAD\\ATV06.RES;1", serializeRes(menu));
    return true;
}
