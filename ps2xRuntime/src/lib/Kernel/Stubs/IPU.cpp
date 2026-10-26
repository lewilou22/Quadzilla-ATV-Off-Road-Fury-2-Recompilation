#include "Common.h"
#include "IPU.h"
#include "runtime/ee_scheduler.h"

namespace
{
    constexpr uint32_t REG_IPU_CTRL = 0x10002010u;
    constexpr uint32_t REG_IPU_CMD = 0x10002000u;
    constexpr uint32_t REG_IPU_IN_FIFO = 0x10007010u;
    constexpr uint32_t IQVAL_BASE = 0x1721e0u;
    constexpr uint32_t VQVAL_BASE = 0x172230u;
    constexpr uint32_t SETD4_CHCR_ENTRY = 0x126428u;

    void completeIpuInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        PS2Memory &mem = runtime->memory();
        mem.write32(REG_IPU_CTRL, 0x40000000u);
        mem.write32(REG_IPU_CMD, 0u);

        __m128i v;
        v = runtime->Load128(rdram, ctx, IQVAL_BASE + 0x00u);
        mem.write128(REG_IPU_IN_FIFO, v);
        v = runtime->Load128(rdram, ctx, IQVAL_BASE + 0x10u);
        mem.write128(REG_IPU_IN_FIFO, v);
        v = runtime->Load128(rdram, ctx, IQVAL_BASE + 0x20u);
        mem.write128(REG_IPU_IN_FIFO, v);
        v = runtime->Load128(rdram, ctx, IQVAL_BASE + 0x30u);
        mem.write128(REG_IPU_IN_FIFO, v);
        v = runtime->Load128(rdram, ctx, IQVAL_BASE + 0x40u);
        mem.write128(REG_IPU_IN_FIFO, v);
        mem.write128(REG_IPU_IN_FIFO, v);
        mem.write128(REG_IPU_IN_FIFO, v);
        mem.write128(REG_IPU_IN_FIFO, v);

        mem.write32(REG_IPU_CMD, 0x50000000u);
        mem.write32(REG_IPU_CMD, 0x58000000u);

        v = runtime->Load128(rdram, ctx, VQVAL_BASE + 0x00u);
        mem.write128(REG_IPU_IN_FIFO, v);
        v = runtime->Load128(rdram, ctx, VQVAL_BASE + 0x10u);
        mem.write128(REG_IPU_IN_FIFO, v);

        mem.write32(REG_IPU_CMD, 0x60000000u);
        mem.write32(REG_IPU_CMD, 0x90000000u);
        mem.write32(REG_IPU_CTRL, 0x40000000u);
        mem.write32(REG_IPU_CMD, 0u);
        setReturnS32(ctx, 0);
    }
}

namespace ps2_stubs
{
    void sceIpuInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
            return;

        if (!runtime->memory().getRDRAM())
        {
            if (!runtime->memory().initialize())
            {
                setReturnS32(ctx, -1);
                return;
            }
        }

        if (!runtime->syncCoreSubsystems())
        {
            setReturnS32(ctx, -1);
            return;
        }

        if (runtime->hasFunction(SETD4_CHCR_ENTRY))
        {
            EeScheduler &scheduler = runtime->eeScheduler();
            scheduler.bindMainContextForSyscall(*ctx, rdram);
            GuestInvocation invocation{};
            invocation.kind = GuestInvocationKind::HleCall;
            invocation.context = *ctx;
            invocation.context.pc = SETD4_CHCR_ENTRY;
            SET_GPR_U32(&invocation.context, 4, 1u);
            SET_GPR_U32(&invocation.context, 29, 0u);
            SET_GPR_U32(&invocation.context, 31, 0u);
            invocation.onComplete = [rdram, runtime](const R5900Context &, R5900Context &parent)
            {
                completeIpuInit(rdram, &parent, runtime);
            };
            scheduler.invokeCurrent(std::move(invocation));
        }

        completeIpuInit(rdram, ctx, runtime);
    }

    void sceIpuRestartDMA(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceIpuStopDMA(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceIpuSync(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }
}

#if !defined(PS2X_HAS_FFMPEG)
#define PS2X_HAS_FFMPEG 1
#endif

#if PS2X_HAS_FFMPEG
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/log.h>
}
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

// Software IDEC for texture streams: the IPU bitstream is rewritten as an MPEG-2 intra
// picture (ported from the ATV1 runtime), decoded with FFmpeg, then colour-converted and laid
// out the way the IPU's FROM FIFO delivers it.
namespace
{
    namespace ipu_idec
    {
        class InBitStream
        {
        public:
            InBitStream(const uint8_t *data, size_t size)
                : m_data(data), m_size(size), m_byteIndex(0), m_bitMask(0x80) {}

            void setPos(size_t bytePos, uint32_t bit)
            {
                m_byteIndex = bytePos;
                m_bitMask = 0x80;
                skipBits(static_cast<int>(bit));
            }

            void getPos(size_t &bytePos, uint32_t &bit) const
            {
                uint32_t mask = m_bitMask;
                uint32_t count = 0;
                while (mask != 0x80)
                {
                    count++;
                    mask = (mask << 1) & 0xFF;
                }
                bit = count;
                bytePos = m_byteIndex;
            }

            uint32_t getBits(int n)
            {
                uint32_t dest = 0;
                for (int i = 0; i < n; ++i)
                {
                    if (m_byteIndex >= m_size)
                        return 0;
                    dest <<= 1;
                    if ((m_data[m_byteIndex] & m_bitMask) != 0)
                        dest |= 1;
                    if (m_bitMask > 1)
                        m_bitMask >>= 1;
                    else
                    {
                        m_byteIndex++;
                        m_bitMask = 0x80;
                    }
                }
                return dest;
            }

            void skipBits(int n)
            {
                for (int i = 0; i < n; ++i)
                {
                    if (m_byteIndex >= m_size)
                        return;
                    if (m_bitMask > 1)
                        m_bitMask >>= 1;
                    else
                    {
                        m_byteIndex++;
                        m_bitMask = 0x80;
                    }
                }
            }

            int getDcsY()
            {
                uint32_t bits = getBits(2);
                if (bits == 0) return 1;
                if (bits == 1) return 2;
                bits = (bits << 1) | getBits(1);
                if (bits == 4) return 0;
                if (bits == 5) return 3;
                if (bits == 6) return 4;
                if (getBits(1) == 0) return 5;
                if (getBits(1) == 0) return 6;
                if (getBits(1) == 0) return 7;
                if (getBits(1) == 0) return 8;
                if (getBits(1) == 0) return 9;
                if (getBits(1) == 0) return 10;
                return 11;
            }

            int getDcsC()
            {
                uint32_t bits = getBits(2);
                if (bits == 0) return 0;
                if (bits == 1) return 1;
                if (bits == 2) return 2;
                if (getBits(1) == 0) return 3;
                if (getBits(1) == 0) return 4;
                if (getBits(1) == 0) return 5;
                if (getBits(1) == 0) return 6;
                if (getBits(1) == 0) return 7;
                if (getBits(1) == 0) return 8;
                if (getBits(1) == 0) return 9;
                if (getBits(1) == 0) return 10;
                return 11;
            }

        private:
            const uint8_t *m_data;
            size_t m_size;
            size_t m_byteIndex;
            uint32_t m_bitMask;
        };

        class OutBitStream
        {
        public:
            OutBitStream() : m_outCnt(0), m_outBuf(0)
            {
                m_data.reserve(32768);
            }

            void putBits(uint32_t data, int n)
            {
                if (n <= 0) return;
                uint32_t mask = 1u << (n - 1);
                for (int i = 0; i < n; ++i)
                {
                    m_outBuf = static_cast<uint8_t>((m_outBuf << 1) & 0xFF);
                    if ((data & mask) != 0)
                        m_outBuf |= 1;
                    mask >>= 1;
                    m_outCnt++;
                    if (m_outCnt == 8)
                    {
                        m_data.push_back(m_outBuf);
                        m_outCnt = 0;
                        m_outBuf = 0;
                    }
                }
            }

            void putBuf()
            {
                if (m_outCnt > 0)
                {
                    putBits(0, 8 - m_outCnt);
                }
            }

            void putDcsY(int len)
            {
                static const struct { uint32_t val; int bits; } lut[] = {
                    {4, 3}, {0, 2}, {1, 2}, {5, 3}, {6, 3}, {14, 4},
                    {30, 5}, {62, 6}, {126, 7}, {254, 8}, {510, 9}, {511, 9}
                };
                if (len >= 0 && len <= 11)
                    putBits(lut[len].val, lut[len].bits);
            }

            void putDcsC(int len)
            {
                static const struct { uint32_t val; int bits; } lut[] = {
                    {0, 2}, {1, 2}, {2, 2}, {6, 3}, {14, 4}, {30, 5},
                    {62, 6}, {126, 7}, {254, 8}, {510, 9}, {1022, 10}, {1023, 10}
                };
                if (len >= 0 && len <= 11)
                    putBits(lut[len].val, lut[len].bits);
            }

            const std::vector<uint8_t>& buffer() const { return m_data; }

        private:
            std::vector<uint8_t> m_data;
            int m_outCnt;
            uint8_t m_outBuf;
        };

        static int vlc(InBitStream &in, OutBitStream &out, bool writeOutput, bool mp1)
        {
            uint32_t bits = in.getBits(2);
            if (writeOutput) out.putBits(bits, 2);

            if (bits == 2) return 1; // EOB
            if (bits == 3) return 0; // token, needs sign

            if (bits == 1)
            {
                bits = in.getBits(1);
                if (writeOutput) out.putBits(bits, 1);
                if (bits != 0) return 0;
                bits = in.getBits(1);
                if (writeOutput) out.putBits(bits, 1);
                return 0;
            }

            bits = in.getBits(1);
            if (writeOutput) out.putBits(bits, 1);

            if (bits != 0)
            {
                bits = in.getBits(2);
                if (writeOutput) out.putBits(bits, 2);
                if (bits < 1)
                {
                    bits = in.getBits(3);
                    if (writeOutput) out.putBits(bits, 3);
                }
                return 0;
            }
            else
            {
                bits = in.getBits(3);
                if (writeOutput) out.putBits(bits, 3);

                if (bits >= 4) return 0;
                if (bits >= 2)
                {
                    bits = in.getBits(1);
                    if (writeOutput) out.putBits(bits, 1);
                    return 0;
                }

                if (bits == 1)
                {
                    if (mp1)
                    {
                        uint32_t run = in.getBits(6);
                        if (writeOutput) out.putBits(run, 6);
                        uint32_t lev = in.getBits(8);
                        if (writeOutput) out.putBits(lev, 8);
                        if (lev == 0 || lev == 0x80)
                        {
                            uint32_t levExt = in.getBits(8);
                            if (writeOutput) out.putBits(levExt, 8);
                        }
                    }
                    else
                    {
                        bits = in.getBits(18);
                        if (writeOutput) out.putBits(bits, 18);
                    }
                    return 2; // escape
                }

                bits = in.getBits(1);
                if (writeOutput) out.putBits(bits, 1);

                if (bits != 0)
                {
                    bits = in.getBits(3);
                    if (writeOutput) out.putBits(bits, 3);
                    return 0;
                }

                int level = 0;
                do
                {
                    bits = in.getBits(1);
                    if (writeOutput) out.putBits(bits, 1);
                    level++;
                } while (bits == 0 && level < 6);

                if (level < 6)
                {
                    bits = in.getBits(4);
                    if (writeOutput) out.putBits(bits, 4);
                    return 0;
                }
                return -1; // invalid VLC
            }
        }

        struct MBData
        {
            size_t bytePos = 0;
            uint32_t bitPos = 0;
            int dct_dc_y = 0;
            int dct_dc_cb = 0;
            int dct_dc_cr = 0;
            int quant = 1;
        };

        static bool convertIpuToM2v(const uint8_t *ipuData, size_t ipuSize,
                                    uint32_t sizex, uint32_t sizey,
                                    std::vector<uint8_t> &outM2v)
        {
            if (!ipuData || ipuSize < 17 || sizex == 0 || sizey == 0)
                return false;

            InBitStream in(ipuData, ipuSize);
            OutBitStream out;

            // Skip 16-byte IPU header if present
            if (in.getBits(32) == 0x6970756du) // "ipum"
            {
                in.getBits(32); // size
                uint32_t w = in.getBits(8) | (in.getBits(8) << 8);
                uint32_t h = in.getBits(8) | (in.getBits(8) << 8);
                in.getBits(32); // frames
                if (w > 0) sizex = w;
                if (h > 0) sizey = h;
            }
            else
            {
                in.setPos(0, 0);
            }

            uint32_t flag = in.getBits(8);
            bool mp1 = ((flag & 128) != 0);

            int mbW = static_cast<int>(sizex / 16);
            int mbH = static_cast<int>(sizey / 16);
            int mbCount = mbW * mbH;
            if (mbCount <= 0 || mbCount > 65536)
                return false;

            std::vector<MBData> mbData(mbCount);

            // Sequence Header (00 00 01 B3)
            out.putBits(0x1B3, 32);
            out.putBits(sizex, 12);
            out.putBits(sizey, 12);
            out.putBits(1, 4); // aspect ratio
            out.putBits(3, 4); // frame rate 25fps (pal)
            out.putBits(0x3FFFF, 18); // bitrate
            out.putBits(1, 1); // marker
            out.putBits(112, 10); // vbv buffer
            out.putBits(0, 1);
            out.putBits(0, 1);
            out.putBits(0, 1);

            if (!mp1)
            {
                out.putBits(0x1B5, 32); // extension start code
                out.putBits(1, 4);      // sequence_extension_id
                out.putBits(4, 4);      // profile_and_level
                out.putBits(8, 4);      // progressive_sequence
                out.putBits(1, 1);      // chroma_format 4:2:0
                out.putBits(1, 2);
                out.putBits(0, 2);
                out.putBits(0, 2);
                out.putBits(0, 12);
                out.putBits(1, 1);
                out.putBits(0, 8);
                out.putBits(0, 1);
                out.putBits(0, 2);
                out.putBits(0, 5);
            }

            // GOP Header (00 00 01 B8)
            out.putBits(0x1B8, 32);
            out.putBits(0, 1); // drop frame
            out.putBits(0, 5); // h
            out.putBits(0, 6); // m
            out.putBits(1, 1); // marker
            out.putBits(0, 6); // s
            out.putBits(0, 6); // p
            out.putBits(1, 1); // closed_gop
            out.putBits(0, 6); // broken_link

            // Picture Header (00 00 01 00)
            out.putBits(0x100, 32);
            out.putBits(0, 10); // temporal ref
            out.putBits(1, 3);  // I picture
            out.putBits(0xFFFF, 16); // vbv delay
            out.putBits(0, 3);

            // Picture Coding Extension
            if (!mp1)
            {
                out.putBits(0x1B5, 32);
                out.putBits(0x8FFFF, 20);
                out.putBits(flag & 3, 2); // intra_dc_precision
                out.putBits(3, 2);        // frame
                out.putBits(2, 3);        // top_field_first
                out.putBits((flag & 64) / 64, 1); // q_scale_type
                out.putBits((flag & 32) / 32, 1); // intra_vlc
                out.putBits((flag & 16) / 16, 1); // alt scan
                out.putBits(1, 2);
                out.putBits(0x80, 8);
            }

            // Pass 1: scan MBs to collect DC predictors & quant
            int dct_dc_y = 0;
            int dct_dc_cb = 0;
            int dct_dc_cr = 0;
            int quant = 1;

            for (int mb = 0; mb < mbCount; ++mb)
            {
                if (mb > 0)
                {
                    in.getBits(1); // MBAI
                }
                in.getPos(mbData[mb].bytePos, mbData[mb].bitPos);

                int intraquant = 0;
                if (in.getBits(1) == 0)
                {
                    in.getBits(1);
                    intraquant = 1;
                }
                if ((flag & 4) != 0)
                {
                    in.getBits(1);
                }
                if (intraquant != 0)
                {
                    quant = static_cast<int>(in.getBits(5));
                }
                mbData[mb].quant = quant;

                for (int block = 0; block < 6; ++block)
                {
                    if (block < 4)
                    {
                        int size = in.getDcsY();
                        if (size != 0)
                        {
                            int diff = static_cast<int>(in.getBits(size));
                            if ((diff & (1 << (size - 1))) == 0)
                            {
                                diff = (-1 << size) | (diff + 1);
                            }
                            dct_dc_y += diff;
                        }
                        if (block == 0)
                            mbData[mb].dct_dc_y = dct_dc_y;
                    }
                    else
                    {
                        int size = in.getDcsC();
                        int diff = 0;
                        if (size != 0)
                        {
                            diff = static_cast<int>(in.getBits(size));
                            if ((diff & (1 << (size - 1))) == 0)
                            {
                                diff = (-1 << size) | (diff + 1);
                            }
                        }
                        if (block == 4)
                        {
                            if (size != 0) dct_dc_cb += diff;
                            mbData[mb].dct_dc_cb = dct_dc_cb;
                        }
                        else
                        {
                            if (size != 0) dct_dc_cr += diff;
                            mbData[mb].dct_dc_cr = dct_dc_cr;
                        }
                    }

                    while (true)
                    {
                        int eob = vlc(in, out, false, mp1);
                        if (eob == 0)
                            in.getBits(1); // sign
                        else if (eob == 1)
                            break;
                        else if (eob < 0)
                            return false;
                    }
                }
            }

            // Pass 2: output MPEG-2 slices (raster order)
            for (int sliceRow = 0; sliceRow < mbH; ++sliceRow)
            {
                dct_dc_y = 0;
                dct_dc_cb = 0;
                dct_dc_cr = 0;
                int firstMbSource = sliceRow * mbW;
                int sliceQsc = mbData[firstMbSource].quant;

                out.putBits(1, 24); // 00 00 01
                out.putBits(static_cast<uint32_t>(sliceRow + 1), 8);
                out.putBits(static_cast<uint32_t>(sliceQsc & 0x1F), 5);
                out.putBits(0, 1);
                quant = sliceQsc;

                for (int mbCol = 0; mbCol < mbW; ++mbCol)
                {
                    int mb_source = sliceRow * mbW + mbCol;
                    out.putBits(1, 1); // MBAI = 1
                    in.setPos(mbData[mb_source].bytePos, mbData[mb_source].bitPos);

                    int intraquant = 0;
                    if (in.getBits(1) == 0)
                    {
                        in.getBits(1);
                        intraquant = 1;
                    }

                    bool writeOutputQuant = (mbData[mb_source].quant != quant);
                    if (writeOutputQuant)
                        out.putBits(1, 2);
                    else
                        out.putBits(1, 1);

                    if ((flag & 4) != 0)
                        out.putBits(in.getBits(1), 1);

                    if (intraquant != 0)
                        in.getBits(5);

                    if (writeOutputQuant)
                    {
                        out.putBits(static_cast<uint32_t>(mbData[mb_source].quant & 0x1F), 5);
                        quant = mbData[mb_source].quant;
                    }

                    for (int block = 0; block < 6; ++block)
                    {
                        if (block == 0)
                        {
                            int skipSize = in.getDcsY();
                            if (skipSize != 0) in.getBits(skipSize);
                            int diff = mbData[mb_source].dct_dc_y - dct_dc_y;
                            dct_dc_y = mbData[mb_source].dct_dc_y;
                            int absval = std::abs(diff);
                            int size = 0;
                            while (absval != 0)
                            {
                                absval >>= 1;
                                size++;
                            }
                            out.putDcsY(size);
                            int codeVal = diff;
                            if (codeVal <= 0) codeVal += (1 << size) - 1;
                            out.putBits(static_cast<uint32_t>(codeVal), size);
                        }
                        else if (block > 3)
                        {
                            int skipSize = in.getDcsC();
                            if (skipSize != 0) in.getBits(skipSize);
                            int diff = (block == 4) ? (mbData[mb_source].dct_dc_cb - dct_dc_cb)
                                                    : (mbData[mb_source].dct_dc_cr - dct_dc_cr);
                            if (block == 4)
                                dct_dc_cb = mbData[mb_source].dct_dc_cb;
                            else
                                dct_dc_cr = mbData[mb_source].dct_dc_cr;
                            int absval = std::abs(diff);
                            int size = 0;
                            while (absval != 0)
                            {
                                absval >>= 1;
                                size++;
                            }
                            out.putDcsC(size);
                            int codeVal = diff;
                            if (codeVal <= 0) codeVal += (1 << size) - 1;
                            out.putBits(static_cast<uint32_t>(codeVal), size);
                        }
                        else
                        {
                            int size = in.getDcsY();
                            out.putDcsY(size);
                            int diff = static_cast<int>(in.getBits(size));
                            out.putBits(static_cast<uint32_t>(diff), size);
                            if (size != 0)
                            {
                                if ((diff & (1 << (size - 1))) == 0)
                                    diff = (-1 << size) | (diff + 1);
                                dct_dc_y += diff;
                            }
                        }

                        while (true)
                        {
                            int eob = vlc(in, out, true, mp1);
                            if (eob == 0)
                                out.putBits(in.getBits(1), 1);
                            else if (eob == 1)
                                break;
                            else if (eob < 0)
                                return false;
                        }
                    }
                }
                out.putBuf();
            }

            // Sequence end code (00 00 01 B7)
            out.putBits(0x1B7, 32);
            outM2v = out.buffer();
            return true;
        }

        int clampByte(int v)
        {
            return v < 0 ? 0 : (v > 255 ? 255 : v);
        }

        // CSC is BT.601 studio swing with chroma replicated over each 2x2 pixel quad. OFM=1
        // output is A1B5G5R5. The game leaves the SETTH thresholds at zero, so every pixel's CSC
        // alpha is 0x80 and A is 0 (A is 1 only for alpha 0x40, as in PCSX2). Black then comes out
        // 0x0000, which TEXA AEM=1 makes transparent: the gaps between the slats of race buildings.
        // Each 16x16 macroblock is emitted as one contiguous 512-byte block, in bitstream order.
        bool decodeM2vToMacroblockRgb16(const std::vector<uint8_t> &m2v, uint32_t width, uint32_t height,
                                        uint8_t *dst)
        {
#if PS2X_HAS_FFMPEG
            const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_MPEG2VIDEO);
            if (!codec)
                return false;
            AVCodecContext *codecCtx = avcodec_alloc_context3(codec);
            if (!codecCtx)
                return false;
            codecCtx->width = static_cast<int>(width);
            codecCtx->height = static_cast<int>(height);
            if (avcodec_open2(codecCtx, codec, nullptr) < 0)
            {
                avcodec_free_context(&codecCtx);
                return false;
            }

            std::vector<uint8_t> padded(m2v.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0u);
            std::memcpy(padded.data(), m2v.data(), m2v.size());
            AVPacket *pkt = av_packet_alloc();
            AVFrame *frame = av_frame_alloc();
            pkt->data = padded.data();
            pkt->size = static_cast<int>(m2v.size());

            bool gotFrame = false;
            if (avcodec_send_packet(codecCtx, pkt) >= 0)
            {
                gotFrame = avcodec_receive_frame(codecCtx, frame) >= 0;
                if (!gotFrame)
                {
                    avcodec_send_packet(codecCtx, nullptr);
                    gotFrame = avcodec_receive_frame(codecCtx, frame) >= 0;
                }
            }
            gotFrame = gotFrame && frame->format == AV_PIX_FMT_YUV420P &&
                       frame->width >= static_cast<int>(width) && frame->height >= static_cast<int>(height);

            if (gotFrame)
            {
                const uint32_t mbW = width / 16u;
                for (uint32_t y = 0; y < height; ++y)
                {
                    const uint8_t *yRow = frame->data[0] + static_cast<size_t>(y) * frame->linesize[0];
                    const uint8_t *cbRow = frame->data[1] + static_cast<size_t>(y / 2u) * frame->linesize[1];
                    const uint8_t *crRow = frame->data[2] + static_cast<size_t>(y / 2u) * frame->linesize[2];
                    for (uint32_t x = 0; x < width; ++x)
                    {
                        const int lum = (static_cast<int>(yRow[x]) - 16) * 149;
                        const int cb = static_cast<int>(cbRow[x / 2u]) - 128;
                        const int cr = static_cast<int>(crRow[x / 2u]) - 128;
                        const int r = clampByte((lum + 204 * cr + 64) >> 7);
                        const int g = clampByte((lum - 104 * cr - 50 * cb + 64) >> 7);
                        const int b = clampByte((lum + 258 * cb + 64) >> 7);
                        const uint16_t px = static_cast<uint16_t>(((b >> 3) << 10) | ((g >> 3) << 5) | (r >> 3));
                        const uint32_t mb = (y / 16u) * mbW + (x / 16u);
                        const uint32_t offset = mb * 512u + ((y & 15u) * 16u + (x & 15u)) * 2u;
                        std::memcpy(dst + offset, &px, sizeof(px));
                    }
                }
            }

            av_frame_free(&frame);
            av_packet_free(&pkt);
            avcodec_free_context(&codecCtx);
            return gotFrame;
#else
            (void)m2v;
            (void)width;
            (void)height;
            (void)dst;
            return false;
#endif
        }
    }
}

// Decodes one IPU intra texture stream (optionally prefixed by its 16-byte "ipum" header) into
// width*height*2 bytes of macroblock-ordered RGB16, as IDEC with OFM=1 would write via IPU_FROM.
bool ps2IpuDecodeTextureRgb16(const uint8_t *src, size_t srcSize, uint32_t width, uint32_t height, uint8_t *dst)
{
    if (!src || !dst || width == 0u || height == 0u || (width & 15u) != 0u || (height & 15u) != 0u)
        return false;
    std::vector<uint8_t> m2v;
    if (!ipu_idec::convertIpuToM2v(src, srcSize, width, height, m2v))
        return false;
    return ipu_idec::decodeM2vToMacroblockRgb16(m2v, width, height, dst);
}
