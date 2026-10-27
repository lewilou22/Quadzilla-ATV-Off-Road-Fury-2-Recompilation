#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "ps2_log.h"
#include "runtime/ps2_memory.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <variant>
#include <vector>

extern uint32_t g_ps2XgkickCount;
bool g_raceLoadStarted = false;
void atvStartHostProfilerForThisThread();

namespace
{
    static constexpr uint32_t kHostFrameWidth = 640u;

    // Opt-in (ATV_TRACE_GS_VIEWPORT=1): bounded log of XYOFFSET/SCISSOR
    // changes with the vertex kicks since the previous change, to check the
    // ordering of viewport packets against the geometry they should clip.
    // Arms on the first VU1 XGKICK after an in-range sub-viewport XYOFFSET.
    bool traceGsViewportEnabled()
    {
        static const bool s_enabled = std::getenv("ATV_TRACE_GS_VIEWPORT") != nullptr;
        return s_enabled;
    }
    // Opt-in (ATV_TRACE_TEXPOOL=afterSeconds): draws sampling a 32x64 texture
    // (the off-track HUD arrow) once afterSeconds have passed since race load,
    // each preceded by the last uploads into the 0x3400..0x3E00 texture pool.
    // Consecutive identical arrow draws are collapsed.
    bool texPoolTraceActive()
    {
        static const char *s_env = std::getenv("ATV_TRACE_TEXPOOL");
        if (!s_env || !g_raceLoadStarted)
            return false;
        using Clock = std::chrono::steady_clock;
        static const Clock::time_point s_raceStart = Clock::now();
        static const double s_after = std::atof(s_env);
        return std::chrono::duration<double>(Clock::now() - s_raceStart).count() >= s_after;
    }
    uint32_t s_texPoolLines = 0u;
    uint32_t s_texPoolLastArrowTbp = 0xFFFFFFFFu;
    uint32_t s_texPoolArrowRun = 0u;
    char s_texPoolUploads[8][96] = {};
    uint32_t s_texPoolUploadCount = 0u;
    uint32_t s_texPoolUploadsSinceArrow = 0u;
    uint64_t s_texPoolUploadSeq = 0u;
    bool s_vpTraceArmed = false;
    bool s_vpSubViewportSeen = false;
    uint32_t s_vpTraceLogs = 0u;
    uint64_t s_vpTraceKicks = 0u;
    float s_vpLastX = 0.0f;
    float s_vpLastY = 0.0f;

    void traceGsViewportKick(float x, float y)
    {
        ++s_vpTraceKicks;
        s_vpLastX = x;
        s_vpLastY = y;
    }

    // Consecutive XGKICK packets are collapsed: the first is logged, the
    // rest are counted and reported before the next logged event.
    uint32_t s_path1Run = 0u;

    void flushPath1Run()
    {
        if (s_path1Run > 1u)
        {
            ++s_vpTraceLogs;
            std::fprintf(stderr, "[gs:vp]   ...GIF-PATH1 run=%u\n", s_path1Run);
        }
        s_path1Run = 0u;
    }

    void traceGsViewportWrite(const char *name, uint64_t oldValue, uint64_t newValue, bool isOffset);
}

// Marker hook for other TUs (DMA starts, render-queue wrappers) so their
// events interleave with the [gs:vp] register trace.  No-op until armed.
void atvTraceGsViewportMarker(const char *label, uint32_t a, uint32_t b)
{
    // Arm on the first VU1 XGKICK after a sub-viewport offset: the menu's
    // slide-in frames before that carry no 3D and would exhaust the budget.
    const bool isPath1 = std::strcmp(label, "GIF-PATH1") == 0;
    if (traceGsViewportEnabled() && !s_vpTraceArmed && s_vpSubViewportSeen && isPath1)
    {
        s_vpTraceArmed = true;
    }
    if (!traceGsViewportEnabled() || !s_vpTraceArmed || s_vpTraceLogs >= 600u)
    {
        return;
    }
    if (isPath1 && s_path1Run != 0u)
    {
        ++s_path1Run;
        return;
    }
    flushPath1Run();
    s_path1Run = isPath1 ? 1u : 0u;
    ++s_vpTraceLogs;
    std::fprintf(stderr, "[gs:vp] %s a=0x%x b=0x%x kicksBefore=%llu\n", label, a, b,
                 static_cast<unsigned long long>(s_vpTraceKicks));
    s_vpTraceKicks = 0u;
}

namespace
{
    void traceGsViewportWrite(const char *name, uint64_t oldValue, uint64_t newValue, bool isOffset)
    {
        if (isOffset && !s_vpTraceArmed)
        {
            // Intro video uses off-screen offsets (OFY > 2048); arm only on an
            // in-range sub-viewport offset such as RiderMenu's 1583,1805.
            const uint32_t ofx = static_cast<uint32_t>((newValue & 0xFFFFu) >> 4);
            const uint32_t ofy = static_cast<uint32_t>(((newValue >> 32) & 0xFFFFu) >> 4);
            if (ofx != 1728u && ofx >= 1024u && ofx <= 2048u && ofy >= 1024u && ofy <= 2048u)
            {
                s_vpSubViewportSeen = true;
            }
        }
        if (!s_vpTraceArmed || oldValue == newValue || s_vpTraceLogs >= 600u)
        {
            return;
        }
        flushPath1Run();
        ++s_vpTraceLogs;
        if (isOffset)
        {
            std::fprintf(stderr, "[gs:vp] %s ofx=%u ofy=%u kicksBefore=%llu last=(%.1f,%.1f)\n", name,
                         static_cast<uint32_t>((newValue & 0xFFFFu) >> 4),
                         static_cast<uint32_t>(((newValue >> 32) & 0xFFFFu) >> 4),
                         static_cast<unsigned long long>(s_vpTraceKicks), s_vpLastX, s_vpLastY);
        }
        else
        {
            std::fprintf(stderr, "[gs:vp] %s x=%u..%u y=%u..%u kicksBefore=%llu last=(%.1f,%.1f)\n", name,
                         static_cast<uint32_t>(newValue & 0x7FFu),
                         static_cast<uint32_t>((newValue >> 16) & 0x7FFu),
                         static_cast<uint32_t>((newValue >> 32) & 0x7FFu),
                         static_cast<uint32_t>((newValue >> 48) & 0x7FFu),
                         static_cast<unsigned long long>(s_vpTraceKicks), s_vpLastX, s_vpLastY);
        }
        s_vpTraceKicks = 0u;
    }

    GSPrimReg decodePrimRegister(uint64_t value)
    {
        GSPrimReg prim{};
        prim.type = static_cast<GSPrimType>(value & 0x7u);
        prim.iip = ((value >> 3) & 1u) != 0u;
        prim.tme = ((value >> 4) & 1u) != 0u;
        prim.fge = ((value >> 5) & 1u) != 0u;
        prim.abe = ((value >> 6) & 1u) != 0u;
        prim.aa1 = ((value >> 7) & 1u) != 0u;
        prim.fst = ((value >> 8) & 1u) != 0u;
        prim.ctxt = ((value >> 9) & 1u) != 0u;
        prim.fix = ((value >> 10) & 1u) != 0u;
        return prim;
    }

    static inline uint64_t loadLE64(const uint8_t *p)
    {
        uint64_t v;
        std::memcpy(&v, p, 8);
        return v;
    }

    struct PackedGifPacketTag
    {
        uint64_t lo = 0u;
        uint64_t hi = 0u;
        uint32_t payloadOffset = 0u;
        uint32_t nloop = 0u;
        uint32_t nreg = 0u;
        uint8_t regs[16]{};
    };

    template <typename Visitor>
    bool visitPackedGifPacket(const uint8_t *data, uint32_t sizeBytes, Visitor &&visitor)
    {
        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            PackedGifPacketTag tag{};
            tag.lo = loadLE64(data + offset);
            tag.hi = loadLE64(data + offset + 8u);

            const uint8_t flg = static_cast<uint8_t>((tag.lo >> 58u) & 0x3u);
            if (flg != GIF_FMT_PACKED)
                return false;

            tag.nloop = static_cast<uint32_t>(tag.lo & 0x7FFFu);
            tag.nreg = static_cast<uint32_t>((tag.lo >> 60u) & 0xFu);
            if (tag.nreg == 0u)
                tag.nreg = 16u;

            const uint64_t payloadBytes64 =
                static_cast<uint64_t>(tag.nloop) * static_cast<uint64_t>(tag.nreg) * 16ull;
            if (payloadBytes64 > 0xFFFFFFFFull)
                return false;

            offset += 16u;
            const uint32_t payloadBytes = static_cast<uint32_t>(payloadBytes64);
            if (payloadBytes > sizeBytes - offset)
                return false;

            tag.payloadOffset = offset;
            for (uint32_t i = 0u; i < tag.nreg; ++i)
                tag.regs[i] = static_cast<uint8_t>((tag.hi >> (i * 4u)) & 0xFu);

            if (!visitor(tag))
                return false;

            offset += payloadBytes;
        }

        return offset == sizeBytes;
    }

    bool validatePackedGifPacket(const uint8_t *data, uint32_t sizeBytes)
    {
        return visitPackedGifPacket(data, sizeBytes, [](const PackedGifPacketTag &)
                                    { return true; });
    }

    std::atomic<uint32_t> s_debugGifPacketCount{0};
    std::atomic<uint32_t> s_debugGsRegisterCount{0};
    std::atomic<uint32_t> s_debugGsPackedVertexCount{0};
    std::atomic<uint32_t> s_debugGsVertexKickCount{0};
    std::atomic<uint32_t> s_debugCopyRegCount{0};
    std::atomic<uint32_t> s_debugTexaWriteCount{0};
    std::atomic<uint32_t> s_debugCvFontUploadCount{0};
    std::atomic<uint32_t> s_debugLocalCopyCount{0};

    // Runs a raster backend on a worker thread so rasterization overlaps
    // EE/VU emulation. Calls that only change GS memory are queued in order.
    // Calls that return data or touch host state wait for the worker to go
    // idle and then run on the caller's thread while holding the queue lock,
    // so the wrapped backend never executes on two threads at once.
    // GetTransferSnapshot is only used for debug views and returns the state
    // as of the last transfer command the worker processed.
    class GSThreadedBackend final : public GSRasterBackend
    {
    public:
        explicit GSThreadedBackend(std::unique_ptr<GSRasterBackend> inner)
            : m_inner(std::move(inner)), m_cpu(dynamic_cast<GSCpuBackend *>(m_inner.get()))
        {
            m_worker = std::thread([this]
                                   { workerLoop(); });
            s_instance.store(this, std::memory_order_release);
        }

        ~GSThreadedBackend() override
        {
            GSThreadedBackend *self = this;
            s_instance.compare_exchange_strong(self, nullptr);
            {
                std::lock_guard<std::mutex> stageLock(m_stageMutex);
                flushStagedLocked(nullptr);
            }
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                waitIdleLocked(lock);
                m_stop = true;
            }
            m_wake.notify_all();
            m_worker.join();
        }

        void Initialize(uint8_t *vram, uint32_t vramSize) override
        {
            syncCall([&]
                     { m_inner->Initialize(vram, vramSize); refreshTransfer(); });
            resetClutMirror();
        }

        void Reset() override
        {
            syncCall([&]
                     { m_inner->Reset(); refreshTransfer(); });
            resetClutMirror();
        }

        void Submit(const GSPrimitiveBatch &batch) override
        {
            std::lock_guard<std::mutex> stageLock(m_stageMutex);
            m_staged.emplace_back(std::in_place_type<GSPrimitiveBatch>, batch);
            if (m_staged.size() >= kStageBatch)
                flushStagedLocked(nullptr);
        }

        void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override
        {
            // Every TEX0/TEX2 write lands here. Drop the loads GSCpuBackend::
            // LoadClut returns from before touching any state: non-indexed
            // PSM, CLD 0/6/7, and CLD 4/5 with CBP equal to the CBP0/CBP1 it
            // holds (mirrored here; only LoadClut and Reset change it).
            if (s_noCmdStage)
            {
                push(Command{ClutCommand{tex0, texclut}});
                return;
            }
            const uint8_t psm = tex0.psm;
            if (psm != GS_PSM_T4 && psm != GS_PSM_T4HL && psm != GS_PSM_T4HH &&
                psm != GS_PSM_T8 && psm != GS_PSM_T8H)
                return;
            std::lock_guard<std::mutex> stageLock(m_stageMutex);
            switch (tex0.cld)
            {
            case 1u:
                break;
            case 2u:
            case 3u:
                m_clutCbpMirror[tex0.cld - 2u] = tex0.cbp;
                break;
            case 4u:
            case 5u:
                if (m_clutCbpMirror[tex0.cld - 4u] == tex0.cbp)
                    return;
                m_clutCbpMirror[tex0.cld - 4u] = tex0.cbp;
                break;
            default:
                return;
            }
            stageLocked(Command{ClutCommand{tex0, texclut}});
        }

        void BeginTransfer(const GSTransferCommand &command) override { push(Command{command}); }

        void UploadImage(const uint8_t *data, uint32_t sizeBytes) override
        {
            UploadCommand upload;
            if (data && sizeBytes != 0u)
                upload.data.assign(data, data + sizeBytes);
            push(Command{std::move(upload)});
        }

        void Flush() override { push(Command{FlushCommand{}}); }
        void TextureFlush() override { push(Command{TextureFlushCommand{}}); }

        void Sync(GSSyncReason reason) override
        {
            // FINISH only latches CSR in the frontend; every VRAM reader
            // (Present, ReadVram, local->host, snapshots) syncs on its own.
            if (reason == GSSyncReason::Finish)
                return;
            syncCall([&]
                     { m_inner->Sync(reason); });
        }

        PresentationFrame Present(const GSPresentationRequest &request) override
        {
            PresentationFrame frame;
            syncCall([&]
                     { frame = m_inner->Present(request); });
            return frame;
        }

        bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override
        {
            bool cleared = false;
            syncCall([&]
                     { cleared = m_inner->ClearFramebuffer(context, rgba); });
            return cleared;
        }

        uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override
        {
            uint32_t consumed = 0u;
            syncCall([&]
                     { consumed = m_inner->ConsumeLocalToHostBytes(dst, maxBytes); refreshTransfer(); });
            return consumed;
        }

        uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override
        {
            uint32_t value = 0u;
            syncCall([&]
                     { value = m_inner->ReadVram(psm, base, bw, x, y); });
            return value;
        }

        void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
        {
            push(Command{WriteVramCommand{psm, base, bw, x, y, value}});
        }

        void SnapshotVram(std::vector<uint8_t> &out) const override
        {
            syncCall([&]
                     { m_inner->SnapshotVram(out); });
        }

        GSTransferSnapshot GetTransferSnapshot() const override
        {
            std::lock_guard<std::mutex> lock(m_transferMutex);
            return m_transfer;
        }

        // Copies the pending local->host bytes into `out` once the worker reaches this point of the
        // command stream, then stores the byte count | 0x80000000 in *done. `out` must stay alive
        // until then. Returns at once.
        void QueueReadback(uint8_t *out, uint32_t maxBytes, std::atomic<uint32_t> *done)
        {
            push(Command{ReadbackCommand{out, maxBytes, done}});
        }

        // The threaded backend in use (gsQueueLocalToHostCopy).
        inline static std::atomic<GSThreadedBackend *> s_instance{nullptr};

    private:
        struct ReadbackCommand
        {
            uint8_t *out;
            uint32_t maxBytes;
            std::atomic<uint32_t> *done;
        };
        struct ClutCommand
        {
            GSTex0Reg tex0;
            GSTexClutReg texclut;
        };
        struct UploadCommand
        {
            std::vector<uint8_t> data;
        };
        struct FlushCommand
        {
        };
        struct TextureFlushCommand
        {
        };
        struct WriteVramCommand
        {
            uint32_t psm, base, bw, x, y, value;
        };
        // Copies local memory at this point of the command stream (see
        // QueueSnapshot); `out` must stay alive until WaitSnapshot returns.
        struct SnapshotCommand
        {
            std::vector<uint8_t> *out;
        };
        using Command = std::variant<GSPrimitiveBatch, ClutCommand, GSTransferCommand, UploadCommand,
                                     FlushCommand, TextureFlushCommand, WriteVramCommand, SnapshotCommand,
                                     ReadbackCommand>;

        static constexpr size_t kMaxQueued = 16384u;
        // Commands are handed to the worker in groups: one queue lock and at
        // most one wake per group instead of per triangle or register write.
        static constexpr size_t kStageBatch = 64u;
        // ATV_GS_NO_CMD_STAGE=1: non-primitive commands flush the stage and
        // take the queue lock each (session 31 behaviour, for A/B).
        inline static const bool s_noCmdStage = []
        {
            const char *env = std::getenv("ATV_GS_NO_CMD_STAGE");
            return env && env[0] != '\0' && env[0] != '0';
        }();

        void resetClutMirror()
        {
            std::lock_guard<std::mutex> stageLock(m_stageMutex);
            m_clutCbpMirror = {};
        }

        // Requires m_stageMutex.
        void stageLocked(Command &&command)
        {
            m_staged.push_back(std::move(command));
            if (m_staged.size() >= kStageBatch)
                flushStagedLocked(nullptr);
        }

        // Ordering is all that matters to the worker; every reader of the
        // worker's results goes through syncCall, which flushes the stage.
        void push(Command &&command)
        {
            std::lock_guard<std::mutex> stageLock(m_stageMutex);
            if (s_noCmdStage)
                flushStagedLocked(&command);
            else
                stageLocked(std::move(command));
        }

        // Requires m_stageMutex. Moves staged commands, then `extra`, into
        // the queue under one m_mutex acquisition, preserving order.
        void flushStagedLocked(Command *extra) const
        {
            if (m_staged.empty() && !extra)
                return;
            if (s_queueStats)
            {
                flushStagedStatsLocked(extra);
                return;
            }
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_queue.size() >= kMaxQueued)
                m_changed.wait(lock, [&]
                               { return m_queue.size() < kMaxQueued; });
            appendStagedLocked(extra);
        }

        // ATV_GS_QUEUE_STATS=1: every ~5 s, how long producers waited for a
        // contended queue lock vs a full queue, and how full the queue was.
        inline static const bool s_queueStats = std::getenv("ATV_GS_QUEUE_STATS") != nullptr;
        mutable uint64_t m_statFlushes = 0, m_statContended = 0, m_statFull = 0, m_statLockNs = 0,
                         m_statFullNs = 0, m_statQueuedSum = 0, m_statNonEmpty = 0, m_statStartNs = 0;

        static uint64_t statNowNs()
        {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count());
        }

        void flushStagedStatsLocked(Command *extra) const
        {
            const uint64_t t0 = statNowNs();
            if (m_statStartNs == 0)
                m_statStartNs = t0;
            std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
            if (!lock.owns_lock())
            {
                lock.lock();
                ++m_statContended;
                m_statLockNs += statNowNs() - t0;
            }
            if (m_queue.size() >= kMaxQueued)
            {
                const uint64_t t1 = statNowNs();
                m_changed.wait(lock, [&]
                               { return m_queue.size() < kMaxQueued; });
                ++m_statFull;
                m_statFullNs += statNowNs() - t1;
            }
            ++m_statFlushes;
            m_statQueuedSum += m_queue.size();
            m_statNonEmpty += m_queue.empty() ? 0u : 1u;
            appendStagedLocked(extra);
            const uint64_t now = statNowNs();
            if (now - m_statStartNs >= 5000000000ull)
            {
                std::fprintf(stderr,
                             "[gs-queue] wall_ms=%.0f flushes=%llu contended=%llu lock_ms=%.1f full=%llu full_ms=%.1f "
                             "avg_queued=%.0f nonempty=%llu race=%d\n",
                             (now - m_statStartNs) / 1e6, static_cast<unsigned long long>(m_statFlushes),
                             static_cast<unsigned long long>(m_statContended), m_statLockNs / 1e6,
                             static_cast<unsigned long long>(m_statFull), m_statFullNs / 1e6,
                             m_statFlushes ? static_cast<double>(m_statQueuedSum) / m_statFlushes : 0.0,
                             static_cast<unsigned long long>(m_statNonEmpty), g_raceLoadStarted ? 1 : 0);
                m_statStartNs = now;
                m_statFlushes = m_statContended = m_statFull = m_statLockNs = m_statFullNs = 0;
                m_statQueuedSum = m_statNonEmpty = 0;
            }
        }

        // Requires m_stageMutex and m_mutex.
        void appendStagedLocked(Command *extra) const
        {
            const bool wasEmpty = m_queue.empty();
            m_pushedSeq += m_staged.size();
            if (wasEmpty)
                m_queue.swap(m_staged); // usual case: no per-command copy
            else
                for (Command &staged : m_staged)
                    m_queue.push_back(std::move(staged));
            m_staged.clear();
            if (extra)
            {
                m_queue.push_back(std::move(*extra));
                ++m_pushedSeq;
            }
            if (wasEmpty && !m_queue.empty())
                m_wake.notify_one();
        }

        void waitIdleLocked(std::unique_lock<std::mutex> &lock) const
        {
            m_changed.wait(lock, [&]
                           { return m_queue.empty() && !m_busy; });
        }

        // Waits only for the commands queued before the call: a producer that
        // never lets the queue go idle (FMV macroblock uploads) would otherwise
        // starve the window thread. Holding m_mutex during fn would block every
        // push from the EE for the whole call (Present copies and converts all
        // of VRAM); the inner backend serializes its own calls.
        template <typename Fn>
        void syncCall(Fn &&fn) const
        {
            {
                std::lock_guard<std::mutex> stageLock(m_stageMutex);
                flushStagedLocked(nullptr);
            }
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                const uint64_t target = m_pushedSeq;
                m_changed.wait(lock, [&]
                               { return m_doneSeq >= target; });
            }
            fn();
        }

    public:
        // Presentation: snapshot local memory after exactly the commands
        // submitted so far (unlike syncCall + SnapshotVram, which reads VRAM
        // while the worker may already run newer commands). QueueSnapshot
        // returns at once; WaitSnapshot waits for that command.
        uint64_t QueueSnapshot(std::vector<uint8_t> &out)
        {
            Command command{SnapshotCommand{&out}};
            std::lock_guard<std::mutex> stageLock(m_stageMutex);
            flushStagedLocked(&command);
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_pushedSeq;
        }
        void WaitSnapshot(uint64_t target) const
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_changed.wait(lock, [&]
                           { return m_doneSeq >= target; });
        }

    private:
        void refreshTransfer() const
        {
            const GSTransferSnapshot snapshot = m_inner->GetTransferSnapshot();
            std::lock_guard<std::mutex> lock(m_transferMutex);
            m_transfer = snapshot;
        }

        void execute(Command &command)
        {
            if (auto *batch = std::get_if<GSPrimitiveBatch>(&command))
                m_inner->Submit(*batch);
            else if (auto *clut = std::get_if<ClutCommand>(&command))
                m_inner->LoadClut(clut->tex0, clut->texclut);
            else if (auto *transfer = std::get_if<GSTransferCommand>(&command))
            {
                m_inner->BeginTransfer(*transfer);
                refreshTransfer();
            }
            else if (auto *upload = std::get_if<UploadCommand>(&command))
            {
                m_inner->UploadImage(upload->data.data(), static_cast<uint32_t>(upload->data.size()));
                refreshTransfer();
            }
            else if (std::holds_alternative<FlushCommand>(command))
                m_inner->Flush();
            else if (std::holds_alternative<TextureFlushCommand>(command))
                m_inner->TextureFlush();
            else if (auto *write = std::get_if<WriteVramCommand>(&command))
                m_inner->WriteVram(write->psm, write->base, write->bw, write->x, write->y, write->value);
            else if (auto *snap = std::get_if<SnapshotCommand>(&command))
                m_inner->SnapshotVram(*snap->out);
            else if (auto *readback = std::get_if<ReadbackCommand>(&command))
            {
                const uint32_t copied = m_inner->ConsumeLocalToHostBytes(readback->out, readback->maxBytes);
                refreshTransfer();
                readback->done->store(copied | 0x80000000u, std::memory_order_release);
            }
        }

        void workerLoop()
        {
            static const bool s_profileWorker = std::getenv("ATV_HOST_PROFILE_GS") != nullptr;
            bool profilerStarted = false;
            std::vector<Command> batch;
            std::vector<const GSPrimitiveBatch *> primitives;
            uint64_t batchEnd = 0;
            uint64_t workerBusyNs = 0, workerCommands = 0, workerBatches = 0, workerStatStart = 0;
            for (;;)
            {
                // ATV_HOST_PROFILE_GS=1 profiles the race; =2 from boot (menus).
                static const bool s_profileWorkerAlways = []
                {
                    const char *env = std::getenv("ATV_HOST_PROFILE_GS");
                    return env && env[0] == '2';
                }();
                if (s_profileWorker && !profilerStarted && (g_raceLoadStarted || s_profileWorkerAlways))
                {
                    profilerStarted = true;
                    atvStartHostProfilerForThisThread();
                }
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_busy = false;
                    m_doneSeq = batchEnd;
                    m_changed.notify_all();
                    m_wake.wait(lock, [&]
                                { return m_stop || !m_queue.empty(); });
                    if (m_queue.empty())
                        return;
                    batch.swap(m_queue);
                    batchEnd = m_pushedSeq;
                    m_busy = true;
                }
                m_changed.notify_all();
                const uint64_t busyStart = s_queueStats ? statNowNs() : 0u;
                // Consecutive primitives go to the CPU backend under one lock.
                // ATV_GS_NO_SUBMIT_MANY=1: one Submit (and lock) per primitive.
                static const bool s_noSubmitMany = std::getenv("ATV_GS_NO_SUBMIT_MANY") != nullptr;
                for (size_t i = 0; i < batch.size();)
                {
                    if (m_cpu && !s_noSubmitMany && std::holds_alternative<GSPrimitiveBatch>(batch[i]))
                    {
                        primitives.clear();
                        for (; i < batch.size(); ++i)
                        {
                            const GSPrimitiveBatch *primitive = std::get_if<GSPrimitiveBatch>(&batch[i]);
                            if (!primitive)
                                break;
                            primitives.push_back(primitive);
                        }
                        m_cpu->SubmitMany(primitives.data(), primitives.size());
                        continue;
                    }
                    execute(batch[i]);
                    ++i;
                }
                if (s_queueStats)
                {
                    const uint64_t now = statNowNs();
                    workerBusyNs += now - busyStart;
                    workerCommands += batch.size();
                    ++workerBatches;
                    if (workerStatStart == 0)
                        workerStatStart = busyStart;
                    if (now - workerStatStart >= 5000000000ull)
                    {
                        std::fprintf(stderr, "[gs-worker] wall_ms=%.0f busy_ms=%.0f batches=%llu commands=%llu\n",
                                     (now - workerStatStart) / 1e6, workerBusyNs / 1e6,
                                     static_cast<unsigned long long>(workerBatches),
                                     static_cast<unsigned long long>(workerCommands));
                        workerStatStart = now;
                        workerBusyNs = workerCommands = workerBatches = 0;
                    }
                }
                batch.clear();
            }
        }

        std::unique_ptr<GSRasterBackend> m_inner;
        GSCpuBackend *m_cpu = nullptr; // m_inner when it is the CPU backend
        // Lock order: m_stageMutex before m_mutex. The worker never takes it.
        mutable std::mutex m_stageMutex;
        mutable std::vector<Command> m_staged;
        std::array<uint32_t, 2> m_clutCbpMirror{};
        mutable std::mutex m_mutex;
        mutable std::condition_variable m_changed;
        mutable std::condition_variable m_wake;
        mutable std::vector<Command> m_queue;
        mutable uint64_t m_pushedSeq = 0;
        uint64_t m_doneSeq = 0;
        bool m_busy = false;
        bool m_stop = false;
        mutable std::mutex m_transferMutex;
        mutable GSTransferSnapshot m_transfer{};
        std::thread m_worker;
    };

    std::unique_ptr<GSRasterBackend> makeDefaultRasterBackend()
    {
        // On by default in ATV2 (ATV1 opts in from play_game.bat); ATV_GS_THREAD=0 draws on the caller.
        static const bool s_threaded = []
        {
            const char *env = std::getenv("ATV_GS_THREAD");
            return env == nullptr || env[0] != '0';
        }();
        auto backend = std::make_unique<GSCpuBackend>();
        if (!s_threaded)
            return backend;
        std::fprintf(stderr, "[gs] threaded raster backend enabled\n");
        return std::make_unique<GSThreadedBackend>(std::move(backend));
    }

    // On by default in ATV2 (ATV_GS_GIF_THREAD=0 parses on the caller; ATV1 opts
    // in from play_game.bat and its GS tests run without it): GS::processGIFPacket only queues a copy of
    // the packet and a dedicated thread parses packets in submission order.
    // Every other public GS entry point (GSApiScope) and the EE GS privileged
    // register accessors (gsSyncAsyncGif) first complete all queued packets on
    // the calling thread, so any observer sees the same state as inline parsing.
    // Lock order: processMutex -> GS::m_stateMutex; queueMutex is a leaf.
    bool asyncGifEnabled()
    {
        static const bool s_enabled = []
        {
            const char *env = std::getenv("ATV_GS_GIF_THREAD");
            return env == nullptr || (env[0] != '\0' && env[0] != '0');
        }();
        return s_enabled;
    }

    // Packet FIFO that keeps its storage: MSVC's std::deque gives every 32-byte vector its own heap
    // block, an allocation and a free per GIF packet on the VIF1/VU1 worker and the GIF thread.
    struct PacketFifo
    {
        std::vector<std::vector<uint8_t>> items;
        size_t head = 0u;
        bool empty() const { return head == items.size(); }
        std::vector<uint8_t> &front() { return items[head]; }
        void push_back(std::vector<uint8_t> &&packet) { items.push_back(std::move(packet)); }
        void pop_front()
        {
            if (++head == items.size())
            {
                items.clear();
                head = 0u;
            }
            else if (head >= 1024u && head * 2u >= items.size())
            {
                items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(head));
                head = 0u;
            }
        }
    };

    struct AsyncGifQueue
    {
        GS *gs = nullptr;
        std::mutex queueMutex;
        std::condition_variable wake;
        PacketFifo queue;
        std::vector<std::vector<uint8_t>> pool;
        bool sleeping = false;
        std::mutex processMutex;
        // Queued plus in-flight packets; reaches zero only after the last
        // packet's GS state changes are complete.
        std::atomic<uint32_t> pending{0u};
    };

    std::atomic<AsyncGifQueue *> s_asyncGif{nullptr};
    std::once_flag s_asyncGifOnce;
    thread_local bool t_gsGifDirect = false;
    thread_local uint32_t t_gsApiDepth = 0u;

    constexpr uint32_t kAsyncGifMaxPending = 4096u;

    // Caller holds processMutex and has t_gsGifDirect set.
    void asyncGifProcessQueued(AsyncGifQueue &q)
    {
        std::vector<uint8_t> packet;
        bool havePacket = false;
        for (;;)
        {
            {
                std::lock_guard<std::mutex> lock(q.queueMutex);
                if (havePacket)
                    q.pool.push_back(std::move(packet));
                if (q.queue.empty())
                    return;
                packet = std::move(q.queue.front());
                q.queue.pop_front();
                havePacket = true;
            }
            q.gs->processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
            q.pending.fetch_sub(1u, std::memory_order_release);
        }
    }

    void asyncGifThreadMain(AsyncGifQueue *q)
    {
        // ATV_HOST_PROFILE_GIF=1 samples this thread (ATV1 made it opt-in too).
        if (std::getenv("ATV_HOST_PROFILE_GIF") != nullptr)
            atvStartHostProfilerForThisThread();
        t_gsGifDirect = true;
        static const uint32_t s_spin = []
        {
            const char *env = std::getenv("ATV_GS_GIF_SPIN");
            return env ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 2000u;
        }();
        for (;;)
        {
            for (uint32_t i = 0u; i < s_spin && q->pending.load(std::memory_order_acquire) == 0u; ++i)
                _mm_pause();
            if (q->pending.load(std::memory_order_acquire) == 0u)
            {
                std::unique_lock<std::mutex> lock(q->queueMutex);
                q->sleeping = true;
                q->wake.wait(lock, [q]
                             { return !q->queue.empty(); });
                q->sleeping = false;
            }
            std::lock_guard<std::mutex> process(q->processMutex);
            asyncGifProcessQueued(*q);
        }
    }

    AsyncGifQueue *asyncGifQueueFor(GS *gs)
    {
        std::call_once(s_asyncGifOnce, [gs]
                       {
            auto *q = new AsyncGifQueue();
            q->gs = gs;
            s_asyncGif.store(q, std::memory_order_release);
            std::thread(asyncGifThreadMain, q).detach();
            std::fprintf(stderr, "[gs] async GIF thread enabled\n"); });
        AsyncGifQueue *q = s_asyncGif.load(std::memory_order_acquire);
        return (q && q->gs == gs) ? q : nullptr;
    }

    // ATV_GIF_DRAIN_STATS=1: every ~5 s, per call site, how often a GS
    // access had to drain queued GIF packets, time spent waiting for the GIF
    // thread to release processMutex, and packets processed inline.
    struct GifDrainStat
    {
        const char *site;
        std::thread::id tid;
        uint64_t drains, waitNs, inlinePackets;
    };
    void gifDrainStatRecord(const char *site, uint64_t waitNs, uint64_t packets)
    {
        static std::mutex s_m;
        static std::vector<GifDrainStat> s_stats;
        static auto s_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(s_m);
        GifDrainStat *st = nullptr;
        const std::thread::id tid = std::this_thread::get_id();
        for (GifDrainStat &e : s_stats)
            if (e.site == site && e.tid == tid)
                st = &e;
        if (!st)
        {
            s_stats.push_back({site, tid, 0, 0, 0});
            st = &s_stats.back();
        }
        ++st->drains;
        st->waitNs += waitNs;
        st->inlinePackets += packets;
        const auto now = std::chrono::steady_clock::now();
        if (now - s_start < std::chrono::seconds(5))
            return;
        s_start = now;
        for (GifDrainStat &e : s_stats)
        {
            if (e.drains)
                std::fprintf(stderr, "[gif-drain] tid=%zu %-36s drains=%llu wait_ms=%.1f inline_pkts=%llu\n",
                             std::hash<std::thread::id>{}(e.tid) % 1000u, e.site,
                             static_cast<unsigned long long>(e.drains), e.waitNs / 1e6,
                             static_cast<unsigned long long>(e.inlinePackets));
            e.drains = e.waitNs = e.inlinePackets = 0;
        }
    }

    // Packets from the VIF1/VU1 worker collect here and enter the queue under
    // one lock (every kGifBatchMax packets, when the worker finishes a work
    // item, or before it drains): the worker and the GIF thread no longer take
    // queueMutex once per packet each. Other threads reach the GS only after
    // fencing the idle worker, so they never miss a batched packet.
    // ATV_GS_GIF_NO_BATCH=1 queues every packet at once.
    constexpr size_t kGifBatchMax = 64u;
    thread_local bool t_gifBatching = false; // set by gsMarkGifBatchingThread
    thread_local std::vector<std::vector<uint8_t>> t_gifBatch;
    thread_local std::vector<std::vector<uint8_t>> t_gifFree;

    void asyncGifFlushLocal(AsyncGifQueue &q)
    {
        if (t_gifBatch.empty())
            return;
        bool notify = false;
        {
            std::lock_guard<std::mutex> lock(q.queueMutex);
            for (std::vector<uint8_t> &packet : t_gifBatch)
                q.queue.push_back(std::move(packet));
            q.pending.fetch_add(static_cast<uint32_t>(t_gifBatch.size()), std::memory_order_relaxed);
            notify = q.sleeping;
            while (!q.pool.empty() && t_gifFree.size() < kGifBatchMax)
            {
                t_gifFree.push_back(std::move(q.pool.back()));
                q.pool.pop_back();
            }
        }
        t_gifBatch.clear();
        if (notify)
            q.wake.notify_one();
    }

    void asyncGifDrain(const char *site = "gsSyncAsyncGif")
    {
        static const bool s_stats = std::getenv("ATV_GIF_DRAIN_STATS") != nullptr;
        AsyncGifQueue *q = s_asyncGif.load(std::memory_order_acquire);
        if (!q || t_gsGifDirect)
            return;
        asyncGifFlushLocal(*q);
        const uint32_t pendingBefore = q->pending.load(std::memory_order_acquire);
        if (pendingBefore == 0u)
            return;
        const auto t0 = s_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::lock_guard<std::mutex> process(q->processMutex);
        const auto t1 = s_stats ? std::chrono::steady_clock::now() : t0;
        const uint32_t pendingLocked = q->pending.load(std::memory_order_acquire);
        t_gsGifDirect = true;
        asyncGifProcessQueued(*q);
        t_gsGifDirect = false;
        if (s_stats)
            gifDrainStatRecord(site, static_cast<uint64_t>(
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                               pendingLocked);
    }

    bool asyncGifEnqueue(GS *gs, const uint8_t *data, uint32_t sizeBytes)
    {
        AsyncGifQueue *q = asyncGifQueueFor(gs);
        if (!q)
            return false;
        if (q->pending.load(std::memory_order_relaxed) >= kAsyncGifMaxPending)
            asyncGifDrain("queueFull");

        static const bool s_noBatch = std::getenv("ATV_GS_GIF_NO_BATCH") != nullptr;
        if (!s_noBatch && t_gifBatching)
        {
            std::vector<uint8_t> packet;
            if (!t_gifFree.empty())
            {
                packet = std::move(t_gifFree.back());
                t_gifFree.pop_back();
            }
            packet.assign(data, data + sizeBytes);
            t_gifBatch.push_back(std::move(packet));
            if (t_gifBatch.size() >= kGifBatchMax)
                asyncGifFlushLocal(*q);
            return true;
        }
        // One lock per packet: copy into this thread's spare buffer, then swap
        // in a recycled buffer for the next packet while pushing.
        thread_local std::vector<uint8_t> t_spare;
        t_spare.assign(data, data + sizeBytes);
        bool notify = false;
        {
            std::lock_guard<std::mutex> lock(q->queueMutex);
            q->queue.push_back(std::move(t_spare));
            q->pending.fetch_add(1u, std::memory_order_relaxed);
            notify = q->sleeping;
            if (!q->pool.empty())
            {
                t_spare = std::move(q->pool.back());
                q->pool.pop_back();
            }
            else
                t_spare = {};
        }
        if (notify)
            q->wake.notify_one();
        return true;
    }

    // Nested GS API calls skip the drain: only the outermost call on a thread
    // may take processMutex, since inner calls already hold GS::m_stateMutex.
    struct GSApiScope
    {
        explicit GSApiScope(const char *site)
        {
            if (t_gsApiDepth++ == 0u)
                asyncGifDrain(site);
        }
        ~GSApiScope() { --t_gsApiDepth; }
        GSApiScope(const GSApiScope &) = delete;
        GSApiScope &operator=(const GSApiScope &) = delete;
    };
}

void gsMarkGifBatchingThread()
{
    t_gifBatching = true;
}

void gsFlushWorkerGif()
{
    if (AsyncGifQueue *q = s_asyncGif.load(std::memory_order_acquire))
        asyncGifFlushLocal(*q);
}

void gsSyncAsyncGif()
{
    if (t_gsApiDepth == 0u)
        asyncGifDrain("privReg");
}

// GS::consumeLocalToHostBytes without the wait: the GIF packets this thread queued (the TRXDIR=1
// transfer) reach the backend first, then the threaded backend copies the transfer's bytes into
// `out` when it gets there and sets *done (byte count | 0x80000000). False, with nothing queued,
// when the raster backend is not threaded.
bool gsQueueLocalToHostCopy(uint8_t *out, uint32_t maxBytes, std::atomic<uint32_t> *done)
{
    GSThreadedBackend *threaded = GSThreadedBackend::s_instance.load(std::memory_order_acquire);
    if (threaded == nullptr)
        return false;
    if (t_gsApiDepth == 0u)
        asyncGifDrain("readback");
    threaded->QueueReadback(out, maxBytes, done);
    return true;
}


GS::GS()
    : m_backend(makeDefaultRasterBackend())
{
    reset();
}

void GS::init(uint8_t *vram, uint32_t vramSize, GSRegisters *privRegs)
{
    GSApiScope apiScope{__func__};
    m_localMemoryStorage = vram;
    m_localMemorySize = vramSize;
    m_privRegs = privRegs;
    if (!m_backend)
        m_backend = makeDefaultRasterBackend();
    m_backend->Initialize(vram, vramSize);
    reset();
}

void GS::reset()
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    std::memset(m_ctx, 0, sizeof(m_ctx));
    m_prim = {};
    m_primRegister = {};
    m_prmodeRegister = {};
    m_curR = 0x80;
    m_curG = 0x80;
    m_curB = 0x80;
    m_curA = 0x80;
    m_curQ = 1.0f;
    m_curS = 0.0f;
    m_curT = 0.0f;
    m_curU = 0;
    m_curV = 0;
    m_curFog = 0;
    m_fogR = 0;
    m_fogG = 0;
    m_fogB = 0;
    m_prmodecont = true;
    m_pabe = false;
    m_scanmsk = 0u;
    m_dimx = 0u;
    m_dthe = 0u;
    m_colclamp = 0u;
    m_texa = {0u, false, 0x80u};
    m_texclut = {0u, 0u, 0u};
    m_bitbltbuf = {};
    m_trxpos = {};
    m_trxreg = {};
    m_trxdir = 3;
    m_vtxCount = 0;
    m_vtxIndex = 0;
    m_preferredDisplaySourceFrame = {};
    m_preferredDisplayDestFbp = 0;
    m_hasPreferredDisplaySource = false;
    if (m_backend)
    {
        m_backend->Flush();
        m_backend->Sync(GSSyncReason::Reset);
        m_backend->Reset();
    }
    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        m_hostPresentationFrame.clear();
        m_hostPresentationWidth = 0u;
        m_hostPresentationHeight = 0u;
        m_hostPresentationDisplayFbp = 0u;
        m_hostPresentationSourceFbp = 0u;
        m_hostPresentationUsedPreferred = false;
        m_hasHostPresentationFrame = false;
    }

    m_debugHistoryWrite = 0;
    m_debugHistoryCount = 0;
    m_debugNextSeq = 1;
    m_debugFrameIndex = 0;
    m_debugLastVsyncTick = UINT64_MAX;

    for (int i = 0; i < 2; ++i)
    {
        m_ctx[i].frame.fbw = 10;
        m_ctx[i].scissor = {0, 639, 0, 447};
        m_ctx[i].xyoffset = {0, 0};
    }
}

GSContext &GS::activeContext()
{
    return m_ctx[m_prim.ctxt ? 1 : 0];
}

void GS::snapshotVRAM()
{
    // Presentation/debug snapshots run outside m_stateMutex so the EE can keep
    // feeding the GS while a backend performs host-side conversion. Keep the
    // selected backend alive and unswappable for the duration of the call.
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    if (!m_backend)
        return;
    std::vector<uint8_t> snapshot;
    m_backend->Sync(GSSyncReason::DebugReadback);
    m_backend->SnapshotVram(snapshot);
    std::lock_guard<std::mutex> lock(m_snapshotMutex);
    m_displaySnapshot.swap(snapshot);
}

const uint8_t *GS::lockDisplaySnapshot(uint32_t &outSize)
{
    m_snapshotMutex.lock();
    if (m_displaySnapshot.empty())
    {
        outSize = 0;
        return nullptr;
    }

    outSize = static_cast<uint32_t>(m_displaySnapshot.size());
    return m_displaySnapshot.data();
}

GSDebugSnapshot GS::getDebugSnapshot() const
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    GSDebugSnapshot snapshot{};
    snapshot.ctx[0] = m_ctx[0];
    snapshot.ctx[1] = m_ctx[1];
    snapshot.prim = m_prim;
    snapshot.texa = m_texa;
    snapshot.texclut = m_texclut;
    snapshot.scanmsk = m_scanmsk;
    snapshot.dimx = m_dimx;
    snapshot.dthe = m_dthe;
    snapshot.colclamp = m_colclamp;
    snapshot.bitbltbuf = m_bitbltbuf;
    snapshot.trxpos = m_trxpos;
    snapshot.trxreg = m_trxreg;
    const GSTransferSnapshot transfer = m_backend ? m_backend->GetTransferSnapshot() : GSTransferSnapshot{};
    snapshot.trxdir = transfer.direction;
    snapshot.transferX = transfer.x;
    snapshot.transferY = transfer.y;
    snapshot.transferTotalPixels = transfer.totalPixels;
    snapshot.transferCopiedPixels = transfer.copiedPixels;
    snapshot.lastDisplayBaseBytes = m_lastDisplayBaseBytes;
    snapshot.preferredDisplaySourceFrame = m_preferredDisplaySourceFrame;
    snapshot.preferredDisplayDestFbp = m_preferredDisplayDestFbp;
    snapshot.hasPreferredDisplaySource = m_hasPreferredDisplaySource;
    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        snapshot.hostPresentationWidth = m_hostPresentationWidth;
        snapshot.hostPresentationHeight = m_hostPresentationHeight;
        snapshot.hostPresentationDisplayFbp = m_hostPresentationDisplayFbp;
        snapshot.hostPresentationSourceFbp = m_hostPresentationSourceFbp;
        snapshot.hostPresentationUsedPreferred = m_hostPresentationUsedPreferred;
        snapshot.hasHostPresentationFrame = m_hasHostPresentationFrame;
    }
    snapshot.localToHostPendingBytes = transfer.localToHostPendingBytes;
    return snapshot;
}

std::vector<GSDebugHistoryEntry> GS::getDebugHistory() const
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    std::vector<GSDebugHistoryEntry> out;
    out.reserve(m_debugHistoryCount);
    const size_t first = (m_debugHistoryWrite + kDebugHistoryCapacity - m_debugHistoryCount) % kDebugHistoryCapacity;
    for (size_t i = 0; i < m_debugHistoryCount; ++i)
    {
        out.push_back(m_debugHistory[(first + i) % kDebugHistoryCapacity]);
    }
    return out;
}

void GS::clearDebugHistory()
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    m_debugHistoryWrite = 0;
    m_debugHistoryCount = 0;
    m_debugNextSeq = 1;
    m_debugFrameIndex = 0;
    m_debugLastVsyncTick = UINT64_MAX;
}

bool GS::isDebugHistoryPaused() const
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_debugHistoryPaused;
}

void GS::setDebugHistoryPaused(bool paused)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    m_debugHistoryPaused = paused;
}

GSDebugHistoryEntry GS::makeDebugEventUnlocked(GSDebugEventKind kind) const
{
    GSDebugHistoryEntry entry{};
    entry.kind = kind;
    entry.prim = m_prim;
    const uint32_t ci = m_prim.ctxt ? 1u : 0u;
    entry.frame = m_ctx[ci].frame;
    entry.zbuf = m_ctx[ci].zbuf;
    entry.tex0 = m_ctx[ci].tex0;
    entry.scissor = m_ctx[ci].scissor;
    entry.test = m_ctx[ci].test;
    entry.alpha = m_ctx[ci].alpha;
    entry.bitbltbuf = m_bitbltbuf;
    entry.trxpos = m_trxpos;
    entry.trxreg = m_trxreg;
    const GSTransferSnapshot transfer = m_backend ? m_backend->GetTransferSnapshot() : GSTransferSnapshot{};
    entry.trxdir = transfer.direction;
    entry.transferPixels = transfer.totalPixels;
    return entry;
}

void GS::recordDebugEventUnlocked(GSDebugHistoryEntry entry)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    const uint64_t tick = m_privRegs ? m_privRegs->vsyncTick.load(std::memory_order_acquire) : 0u;
    if (m_debugLastVsyncTick == UINT64_MAX)
    {
        m_debugLastVsyncTick = tick;
    }
    else if (tick != m_debugLastVsyncTick)
    {
        ++m_debugFrameIndex;
        m_debugLastVsyncTick = tick;
    }

    entry.seq = m_debugNextSeq++;
    entry.vsyncTick = tick;
    entry.frameIndex = m_debugFrameIndex;

    m_debugHistory[m_debugHistoryWrite] = entry;
    m_debugHistoryWrite = (m_debugHistoryWrite + 1u) % kDebugHistoryCapacity;
    if (m_debugHistoryCount < kDebugHistoryCapacity)
    {
        ++m_debugHistoryCount;
    }
}

void GS::recordGifTagDebugEventUnlocked(uint32_t sizeBytes, uint32_t nloop, uint8_t flg, uint32_t nreg)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::GifTag);
    entry.gifSizeBytes = sizeBytes;
    entry.gifNloop = nloop;
    entry.gifFlg = flg;
    entry.gifNreg = static_cast<uint8_t>(std::min<uint32_t>(nreg, 16u));
    recordDebugEventUnlocked(entry);
}

void GS::recordRegisterDebugEventUnlocked(uint8_t regAddr, uint64_t value)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    switch (regAddr)
    {
    case GS_REG_PRIM:
    case GS_REG_TEX0_1:
    case GS_REG_TEX0_2:
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    case GS_REG_TEXA:
    case GS_REG_TEXCLUT:
    case GS_REG_FRAME_1:
    case GS_REG_FRAME_2:
    case GS_REG_ZBUF_1:
    case GS_REG_ZBUF_2:
    case GS_REG_ALPHA_1:
    case GS_REG_ALPHA_2:
    case GS_REG_TEST_1:
    case GS_REG_TEST_2:
    case GS_REG_SCISSOR_1:
    case GS_REG_SCISSOR_2:
    case GS_REG_XYOFFSET_1:
    case GS_REG_XYOFFSET_2:
    case GS_REG_BITBLTBUF:
    case GS_REG_TRXPOS:
    case GS_REG_TRXREG:
    case GS_REG_TRXDIR:
        break;
    default:
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Register);
    entry.reg = regAddr;
    entry.regValue = value;
    recordDebugEventUnlocked(entry);
}

void GS::recordDrawDebugEventUnlocked(int vertexCount)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    if (vertexCount <= 0)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Draw);
    entry.vertexCount = static_cast<uint32_t>(vertexCount);

    const int count = std::min(vertexCount, kMaxVerts);
    entry.xMin = entry.xMax = m_vtxQueue[0].x;
    entry.yMin = entry.yMax = m_vtxQueue[0].y;
    entry.zMin = entry.zMax = m_vtxQueue[0].z;
    entry.aMin = entry.aMax = m_vtxQueue[0].a;

    for (int i = 1; i < count; ++i)
    {
        const GSVertex &v = m_vtxQueue[i];
        entry.xMin = std::min(entry.xMin, v.x);
        entry.xMax = std::max(entry.xMax, v.x);
        entry.yMin = std::min(entry.yMin, v.y);
        entry.yMax = std::max(entry.yMax, v.y);
        entry.zMin = std::min(entry.zMin, v.z);
        entry.zMax = std::max(entry.zMax, v.z);
        entry.aMin = std::min(entry.aMin, v.a);
        entry.aMax = std::max(entry.aMax, v.a);
    }

    recordDebugEventUnlocked(entry);
}

void GS::recordTransferDebugEventUnlocked()
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Transfer);
    entry.transferPixels = m_backend ? m_backend->GetTransferSnapshot().totalPixels : 0u;
    recordDebugEventUnlocked(entry);
}

void GS::recordPresentDebugEventUnlocked(uint32_t displayFbp, uint32_t sourceFbp, uint32_t width, uint32_t height, bool usedPreferred)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Present);
    entry.displayFbp = displayFbp;
    entry.sourceFbp = sourceFbp;
    entry.width = width;
    entry.height = height;
    entry.usedPreferred = usedPreferred;
    recordDebugEventUnlocked(entry);
}

bool GS::getPreferredDisplaySource(GSFrameReg &outSource, uint32_t &outDestFbp) const
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!m_hasPreferredDisplaySource)
    {
        outSource = {};
        outDestFbp = 0u;
        return false;
    }

    outSource = m_preferredDisplaySourceFrame;
    outDestFbp = m_preferredDisplayDestFbp;
    return true;
}

void GS::unlockDisplaySnapshot()
{
    m_snapshotMutex.unlock();
}

uint32_t GS::getLastDisplayBaseBytes() const
{
    GSApiScope apiScope{__func__};
    return m_lastDisplayBaseBytes;
}

void GS::refreshDisplaySnapshot()
{
    GSApiScope apiScope{__func__};
    snapshotVRAM();
}

GSPresentationRequest GS::buildPresentationRequestUnlocked() const
{
    GSPresentationRequest request{};
    if (!m_privRegs)
        return request;
    request.pmode = m_privRegs->pmode;
    request.smode2 = m_privRegs->smode2;
    request.dispfb1 = m_privRegs->dispfb1;
    request.display1 = m_privRegs->display1;
    request.dispfb2 = m_privRegs->dispfb2;
    request.display2 = m_privRegs->display2;
    request.bgcolor = m_privRegs->bgcolor;
    request.vsyncTick = m_privRegs->vsyncTick.load(std::memory_order_acquire);
    request.contextFrames[0] = m_ctx[0].frame;
    request.contextFrames[1] = m_ctx[1].frame;
    request.preferredSource = m_preferredDisplaySourceFrame;
    request.preferredDestFbp = m_preferredDisplayDestFbp;
    request.hasPreferredSource = m_hasPreferredDisplaySource;
    return request;
}

void GS::latchHostPresentationFrame()
{
    GSApiScope apiScope{__func__};
    GSPresentationRequest request{};
    // The display registers and the local-memory snapshot must describe the
    // same point of the GS command stream. Taking them in separate steps let
    // the game flip DISPFB and start its next frame (clear, sky, terrain) in
    // the buffer the request still named, which showed up as half-drawn
    // frames without rider or HUD. m_stateMutex keeps GIF parsing (and so
    // every new draw) out while the request is built and the snapshot is
    // queued at that point of the backend's command stream; waiting for the
    // snapshot and the display conversion run without it.
    thread_local std::vector<uint8_t> snapshot;
    snapshot.clear();
    std::unique_lock<std::mutex> backendLock;
    GSThreadedBackend *threaded = nullptr;
    uint64_t snapshotSeq = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
        if (!m_backend || !m_privRegs)
        {
            std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
            m_hostPresentationFrame.clear();
            m_hasHostPresentationFrame = false;
            m_hostPresentationWidth = m_hostPresentationHeight = 0u;
            return;
        }
        request = buildPresentationRequestUnlocked();
        backendLock = std::unique_lock<std::mutex>(m_backendLifetimeMutex);
        threaded = dynamic_cast<GSThreadedBackend *>(m_backend.get());
        if (threaded)
            snapshotSeq = threaded->QueueSnapshot(snapshot);
        else if (m_backend)
        {
            m_backend->Flush();
            m_backend->Sync(GSSyncReason::Presentation);
            m_backend->SnapshotVram(snapshot);
        }
    }
    if (threaded)
        threaded->WaitSnapshot(snapshotSeq);
    backendLock = {};

    PresentationFrame frame = GSCpuBackend::PresentSnapshot(snapshot, request);

    const bool hasFrame = static_cast<bool>(frame);
    const uint32_t displayFbp = frame.displayFbp;
    const uint32_t sourceFbp = frame.sourceFbp;
    const uint32_t width = frame.width;
    const uint32_t height = frame.height;
    const bool usedPreferred = frame.usedPreferred;
    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        m_hostPresentationFrame = std::move(frame.pixels);
        m_hostPresentationWidth = width;
        m_hostPresentationHeight = height;
        m_hostPresentationDisplayFbp = displayFbp;
        m_hostPresentationSourceFbp = sourceFbp;
        m_hostPresentationUsedPreferred = usedPreferred;
        m_hasHostPresentationFrame = hasFrame;
    }

    if (hasFrame)
    {
        std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
        recordPresentDebugEventUnlocked(displayFbp, sourceFbp, width, height, usedPreferred);
    }
}

bool GS::copyLatchedHostPresentationFrame(std::vector<uint8_t> &outPixels,
                                          uint32_t &outWidth,
                                          uint32_t &outHeight,
                                          uint32_t *outDisplayFbp,
                                          uint32_t *outSourceFbp,
                                          bool *outUsedPreferred) const
{
    std::lock_guard<std::mutex> lock(m_presentationMutex);
    if (!m_hasHostPresentationFrame || m_hostPresentationFrame.empty())
    {
        outPixels.clear();
        outWidth = 0u;
        outHeight = 0u;
        if (outDisplayFbp)
            *outDisplayFbp = 0u;
        if (outSourceFbp)
            *outSourceFbp = 0u;
        if (outUsedPreferred)
            *outUsedPreferred = false;
        return false;
    }

    outWidth = m_hostPresentationWidth;
    outHeight = m_hostPresentationHeight;
    if (outDisplayFbp)
        *outDisplayFbp = m_hostPresentationDisplayFbp;
    if (outSourceFbp)
        *outSourceFbp = m_hostPresentationSourceFbp;
    if (outUsedPreferred)
        *outUsedPreferred = m_hostPresentationUsedPreferred;

    const size_t packedRowBytes = static_cast<size_t>(outWidth) * 4u;
    outPixels.resize(packedRowBytes * static_cast<size_t>(outHeight));
    if (outWidth != 0u && outHeight != 0u)
    {
        const size_t sourceRowBytes = static_cast<size_t>(kHostFrameWidth) * 4u;
        for (uint32_t y = 0; y < outHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * sourceRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * packedRowBytes;
            if (srcOffset + packedRowBytes > m_hostPresentationFrame.size() ||
                dstOffset + packedRowBytes > outPixels.size())
            {
                outPixels.clear();
                outWidth = 0u;
                outHeight = 0u;
                if (outDisplayFbp)
                    *outDisplayFbp = 0u;
                if (outSourceFbp)
                    *outSourceFbp = 0u;
                if (outUsedPreferred)
                    *outUsedPreferred = false;
                return false;
            }

            std::memcpy(outPixels.data() + dstOffset,
                        m_hostPresentationFrame.data() + srcOffset,
                        packedRowBytes);
        }
    }
    return true;
}

void GS::processGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (asyncGifEnabled() && !t_gsGifDirect && data && sizeBytes >= 16u && asyncGifEnqueue(this, data, sizeBytes))
        return;
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!data || sizeBytes < 16 || !m_backend)
        return;

    if (tryProcessNativeImageUploadPacket(data, sizeBytes))
        return;

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t packetIndex = s_debugGifPacketCount.fetch_add(1, std::memory_order_relaxed);
        if (packetIndex < 48u)
        {
            const uint64_t tagLo = loadLE64(data);
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            RUNTIME_LOG("[gs:gif] idx=" << packetIndex
                                        << " size=" << sizeBytes
                                        << " nloop=" << nloop
                                        << " flg=" << static_cast<uint32_t>(flg)
                                        << " nreg=" << nreg
                                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                                        << std::endl);
        }
    });

    uint32_t offset = 0;
    while (offset + 16 <= sizeBytes)
    {
        uint64_t tagLo = loadLE64(data + offset);
        uint64_t tagHi = loadLE64(data + offset + 8);
        offset += 16;

        m_curQ = 1.0f;

        uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFF);
        uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xF);
        if (nreg == 0)
            nreg = 16;

        recordGifTagDebugEventUnlocked(sizeBytes, nloop, flg, nreg);

        bool pre = ((tagLo >> 46) & 1) != 0;
        if (pre)
        {
            writeRegisterUnlocked(GS_REG_PRIM, (tagLo >> 47) & 0x7FF);
        }

        uint8_t regs[16];
        for (uint32_t i = 0; i < nreg; ++i)
            regs[i] = static_cast<uint8_t>((tagHi >> (i * 4)) & 0xF);

        if (flg == GIF_FMT_PACKED)
        {
            for (uint32_t loop = 0; loop < nloop; ++loop)
            {
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 16 > sizeBytes)
                        return;
                    uint64_t lo = loadLE64(data + offset);
                    uint64_t hi = loadLE64(data + offset + 8);
                    offset += 16;
                    writeRegisterPacked(regs[r], lo, hi);
                }
            }
        }
        else if (flg == GIF_FMT_REGLIST)
        {
            for (uint32_t loop = 0; loop < nloop; ++loop)
            {
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 8 > sizeBytes)
                        return;
                    writeRegisterUnlocked(regs[r], loadLE64(data + offset));
                    offset += 8;
                }
            }
            if ((nloop * nreg) & 1)
                offset += 8;
        }
        else if (flg == GIF_FMT_IMAGE)
        {
            uint32_t imageBytes = nloop * 16;
            if (offset + imageBytes > sizeBytes)
                imageBytes = sizeBytes - offset;
            processImageData(data + offset, imageBytes);
            offset += imageBytes;
        }
    }
}

bool GS::processNativePackedGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!data || sizeBytes < 16u || !m_backend)
        return false;

    if (!validatePackedGifPacket(data, sizeBytes))
        return false;

    const bool processed = visitPackedGifPacket(data, sizeBytes, [&](const PackedGifPacketTag &tag)
                                                {
        m_curQ = 1.0f;

        recordGifTagDebugEventUnlocked(sizeBytes, tag.nloop, GIF_FMT_PACKED, tag.nreg);

        const bool pre = ((tag.lo >> 46u) & 1u) != 0u;
        if (pre)
            writeRegisterUnlocked(GS_REG_PRIM, (tag.lo >> 47u) & 0x7FFu);

        uint32_t offset = tag.payloadOffset;
        for (uint32_t loop = 0u; loop < tag.nloop; ++loop)
        {
            for (uint32_t r = 0u; r < tag.nreg; ++r)
            {
                const uint64_t lo = loadLE64(data + offset);
                const uint64_t hi = loadLE64(data + offset + 8u);
                offset += 16u;
                writeRegisterPacked(tag.regs[r], lo, hi);
            }
        }

        return true; });

    if (!processed)
        return false;

    ++m_nativePackedGIFPacketCount;
    return true;
}

void GS::uploadImageNative(uint64_t bitbltbuf,
                           uint64_t trxpos,
                           uint64_t trxreg,
                           uint64_t trxdir,
                           const uint8_t *data,
                           uint32_t sizeBytes)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    uploadImageNativeUnlocked(bitbltbuf, trxpos, trxreg, trxdir, data, sizeBytes);
}

void GS::uploadImageNativeUnlocked(uint64_t bitbltbuf,
                                   uint64_t trxpos,
                                   uint64_t trxreg,
                                   uint64_t trxdir,
                                   const uint8_t *data,
                                   uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0 || !m_backend)
        return;

    writeRegisterUnlocked(GS_REG_BITBLTBUF, bitbltbuf);
    writeRegisterUnlocked(GS_REG_TRXPOS, trxpos);
    writeRegisterUnlocked(GS_REG_TRXREG, trxreg);
    writeRegisterUnlocked(GS_REG_TRXDIR, trxdir);
    processImageData(data, sizeBytes);
    ++m_nativeImageUploadCount;
}

bool GS::tryProcessNativeImageUploadPacket(const uint8_t *data, uint32_t sizeBytes)
{
    constexpr uint32_t kSetupRegisters = 4u;
    constexpr uint32_t kPackedAdPayloadBytes = kSetupRegisters * 16u;
    constexpr uint64_t kPackedAdDescriptor = 0x0Eull;

    if (!data || sizeBytes < 16u + kPackedAdPayloadBytes + 16u)
        return false;

    const uint64_t setupTagLo = loadLE64(data);
    const uint64_t setupTagHi = loadLE64(data + 8u);
    const uint32_t setupNloop = static_cast<uint32_t>(setupTagLo & 0x7FFFu);
    const uint8_t setupFlg = static_cast<uint8_t>((setupTagLo >> 58u) & 0x3u);
    uint32_t setupNreg = static_cast<uint32_t>((setupTagLo >> 60u) & 0xFu);
    if (setupNreg == 0u)
        setupNreg = 16u;

    if (setupNloop != kSetupRegisters ||
        setupFlg != GIF_FMT_PACKED ||
        setupNreg != 1u ||
        (setupTagHi & 0xFull) != kPackedAdDescriptor)
    {
        return false;
    }

    uint64_t regs[kSetupRegisters] = {};
    uint32_t offset = 16u;
    constexpr uint8_t expectedRegs[kSetupRegisters] = {
        GS_REG_BITBLTBUF,
        GS_REG_TRXPOS,
        GS_REG_TRXREG,
        GS_REG_TRXDIR,
    };

    for (uint32_t i = 0; i < kSetupRegisters; ++i)
    {
        regs[i] = loadLE64(data + offset);
        const uint64_t reg = loadLE64(data + offset + 8u);
        if ((reg & 0xFFu) != expectedRegs[i])
            return false;
        offset += 16u;
    }

    const uint32_t trxdirMode = static_cast<uint32_t>(regs[3] & 0x3ull);
    const uint32_t rrw = static_cast<uint32_t>(regs[2] & 0xFFFull);
    const uint32_t rrh = static_cast<uint32_t>((regs[2] >> 32u) & 0xFFFull);
    if (trxdirMode != 0u || rrw == 0u || rrh == 0u)
        return false;

    if (offset + 16u > sizeBytes)
        return false;

    const uint64_t imageTagLo = loadLE64(data + offset);
    const uint8_t imageFlg = static_cast<uint8_t>((imageTagLo >> 58u) & 0x3u);
    const uint32_t imageNloop = static_cast<uint32_t>(imageTagLo & 0x7FFFu);
    if (imageFlg != GIF_FMT_IMAGE || imageNloop == 0u)
        return false;

    offset += 16u;
    const uint64_t imageBytes64 = static_cast<uint64_t>(imageNloop) * 16ull;
    if (imageBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);
    if (offset + imageBytes != sizeBytes)
        return false;

    uploadImageNativeUnlocked(regs[0], regs[1], regs[2], regs[3], data + offset, imageBytes);
    return true;
}

void GS::writeRegisterPacked(uint8_t regDesc, uint64_t lo, uint64_t hi)
{
    switch (regDesc)
    {
    case 0x00:
        writeRegisterUnlocked(GS_REG_PRIM, lo & 0x7FF);
        break;
    case 0x01:
        m_curR = static_cast<uint8_t>(lo & 0xFF);
        m_curG = static_cast<uint8_t>((lo >> 32) & 0xFF);
        m_curB = static_cast<uint8_t>(hi & 0xFF);
        m_curA = static_cast<uint8_t>((hi >> 32) & 0xFF);
        break;
    case 0x02:
    {
        uint32_t sBits = static_cast<uint32_t>(lo & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((lo >> 32) & 0xFFFFFFFF);
        uint32_t qBits = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case 0x03:
        m_curU = static_cast<uint16_t>(lo & 0x3FFFu);
        m_curV = static_cast<uint16_t>((lo >> 32) & 0x3FFFu);
        break;
    case 0x04:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        // GIF PACKED XYZF: Z occupies bits 68..91, F bits 100..107.
        // Unlike the 64-bit A+D register layout, each starts after 4 padding bits.
        uint32_t z = static_cast<uint32_t>((hi >> 4) & 0xFFFFFF);
        uint8_t f = static_cast<uint8_t>((hi >> 36) & 0xFF);
        bool adk = ((hi >> 47) & 1) != 0;
        if (g_raceLoadStarted)
        {
            static uint32_t s_xyzfLogs = 0u;
            if (s_xyzfLogs < 32u)
            {
                std::fprintf(stderr,
                    "[gs:xyzf2] n=%u lo=%016llx hi=%016llx x=%u y=%u z=%u fog=%u adk=%d\n",
                    s_xyzfLogs, lo, hi, x, y, z, static_cast<uint32_t>(f), adk ? 1 : 0);
                ++s_xyzfLogs;
            }
        }
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf] idx=" << debugIndex
                                                    << " x=" << x
                                                    << " y=" << y
                                                    << " z=0x" << std::hex << z
                                                    << std::dec
                                                    << " fog=" << static_cast<uint32_t>(f)
                                                    << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = f;
        vertexKick(!adk);
        break;
    }
    case 0x05:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        uint32_t z = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        bool adk = ((hi >> 47) & 1) != 0;
        if (g_raceLoadStarted)
        {
            static uint32_t s_xyz2Logs = 0u;
            if (s_xyz2Logs < 32u)
            {
                std::fprintf(stderr,
                    "[gs:xyz2] n=%u lo=%016llx hi=%016llx x=%u y=%u z=%u curFog=%u adk=%d\n",
                    s_xyz2Logs, lo, hi, x, y, z, static_cast<uint32_t>(m_curFog), adk ? 1 : 0);
                ++s_xyz2Logs;
            }
        }
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz] idx=" << debugIndex
                                                   << " x=" << x
                                                   << " y=" << y
                                                   << " z=0x" << std::hex << z
                                                   << std::dec
                                                   << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                   << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                   << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(!adk);
        break;
    }
    case 0x0A:
    {
        m_curFog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        if (g_raceLoadStarted)
        {
            static uint32_t s_fogLogs = 0u;
            if (s_fogLogs < 32u)
            {
                std::fprintf(stderr, "[gs:fog] n=%u lo=%016llx hi=%016llx newFog=%u\n",
                    s_fogLogs, lo, hi, static_cast<uint32_t>(m_curFog));
                ++s_fogLogs;
            }
        }
        break;
    }
    case 0x0C:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf3] idx=" << debugIndex
                                                     << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                     << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                     << " kick=0"
                                                     << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                     << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>((hi >> 4) & 0xFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        if (g_raceLoadStarted)
        {
            static uint32_t s_xyzf3Logs = 0u;
            if (s_xyzf3Logs < 32u)
            {
                std::fprintf(stderr, "[gs:xyzf3] n=%u lo=%016llx hi=%016llx fog=%u\n",
                    s_xyzf3Logs, lo, hi, static_cast<uint32_t>(vtx.fog));
                ++s_xyzf3Logs;
            }
        }
        vertexKick(false);
        break;
    }
    case 0x0D:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz3] idx=" << debugIndex
                                                    << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                    << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                    << " kick=0"
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>(hi & 0xFFFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(false);
        break;
    }
    case 0x0E:
    {
        uint8_t addr = static_cast<uint8_t>(hi & 0xFF);
        writeRegisterUnlocked(addr, lo);
        break;
    }
    case 0x0F:
        break;
    default:
        writeRegisterUnlocked(regDesc, lo);
        break;
    }
}

void GS::writeRegister(uint8_t regAddr, uint64_t value)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    writeRegisterUnlocked(regAddr, value);
}

void GS::writeRegisterUnlocked(uint8_t regAddr, uint64_t value)
{
    const bool interestingReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_RGBAQ ||
        regAddr == GS_REG_ST ||
        regAddr == GS_REG_UV ||
        regAddr == GS_REG_XYZ2 ||
        regAddr == GS_REG_XYZ3 ||
        regAddr == GS_REG_XYZF2 ||
        regAddr == GS_REG_XYZF3 ||
        regAddr == GS_REG_TEX0_1 ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX2_1 ||
        regAddr == GS_REG_TEX2_2 ||
        regAddr == GS_REG_TEXCLUT ||
        regAddr == GS_REG_TEXA ||
        regAddr == GS_REG_XYOFFSET_1 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_1 ||
        regAddr == GS_REG_SCISSOR_2 ||
        regAddr == GS_REG_FRAME_1 ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_ALPHA_1 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_1 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_BITBLTBUF ||
        regAddr == GS_REG_TRXPOS ||
        regAddr == GS_REG_TRXREG ||
        regAddr == GS_REG_TRXDIR;

    PS2_IF_AGRESSIVE_LOGS({
        if (interestingReg)
        {
            const uint32_t debugIndex = s_debugGsRegisterCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 128u)
            {
                RUNTIME_LOG("[gs:reg] idx=" << debugIndex
                                            << " reg=0x" << std::hex << static_cast<uint32_t>(regAddr)
                                            << " value=0x" << value
                                            << std::dec
                                            << std::endl);
            }
        }
    });

    const bool isCopyRelevantReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX1_2 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_PABE ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_2;
    PS2_IF_AGRESSIVE_LOGS({
        if (isCopyRelevantReg &&
            s_debugCopyRegCount.fetch_add(1u, std::memory_order_relaxed) < 64u)
        {
            RUNTIME_LOG("[gs:copy-reg] reg=0x"
                        << std::hex << static_cast<uint32_t>(regAddr)
                        << " value=0x" << value
                        << std::dec
                        << " primCtxt=" << static_cast<uint32_t>(m_prim.ctxt)
                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                        << std::endl);
        }
    });

    switch (regAddr)
    {
    case GS_REG_PRIM:
    {
        m_primRegister = decodePrimRegister(value);
        if (m_prmodecont)
        {
            m_prim = m_primRegister;
        }
        else
        {
            // PRIM always selects the primitive topology. With AC=0, all
            // rendering attributes remain sourced from PRMODE.
            m_prim.type = m_primRegister.type;
        }
        m_vtxCount = 0;
        m_vtxIndex = 0;
        break;
    }
    case GS_REG_RGBAQ:
    {
        m_curR = static_cast<uint8_t>(value & 0xFF);
        m_curG = static_cast<uint8_t>((value >> 8) & 0xFF);
        m_curB = static_cast<uint8_t>((value >> 16) & 0xFF);
        m_curA = static_cast<uint8_t>((value >> 24) & 0xFF);
        uint32_t qBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case GS_REG_ST:
    {
        uint32_t sBits = static_cast<uint32_t>(value & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        break;
    }
    case GS_REG_UV:
    {
        m_curU = static_cast<uint16_t>(value & 0x3FFFu);
        m_curV = static_cast<uint16_t>((value >> 16) & 0x3FFFu);
        break;
    }
    case GS_REG_XYZF2:
    case GS_REG_XYZF3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<double>((value >> 32) & 0xFFFFFF);
        vtx.fog = static_cast<uint8_t>((value >> 56) & 0xFF);
        if (traceGsViewportEnabled())
            traceGsViewportKick(vtx.x, vtx.y);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vertexKick(regAddr == GS_REG_XYZF2);
        break;
    }
    case GS_REG_XYZ2:
    case GS_REG_XYZ3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<double>((value >> 32) & 0xFFFFFFFF);
        if (traceGsViewportEnabled())
            traceGsViewportKick(vtx.x, vtx.y);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(regAddr == GS_REG_XYZ2);
        break;
    }
    case GS_REG_TEX0_1:
    case GS_REG_TEX0_2:
    {
        int ci = (regAddr == GS_REG_TEX0_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.tbp0 = static_cast<uint32_t>(value & 0x3FFF);
        t.tbw = static_cast<uint8_t>((value >> 14) & 0x3F);
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.tw = static_cast<uint8_t>((value >> 26) & 0xF);
        t.th = static_cast<uint8_t>((value >> 30) & 0xF);
        t.tcc = static_cast<uint8_t>((value >> 34) & 0x1);
        t.tfx = static_cast<uint8_t>((value >> 35) & 0x3);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        m_backend->LoadClut(t, m_texclut);
        break;
    }
    case GS_REG_CLAMP_1:
    case GS_REG_CLAMP_2:
    {
        int ci = (regAddr == GS_REG_CLAMP_2) ? 1 : 0;
        m_ctx[ci].clamp = value;
        break;
    }
    case GS_REG_FOG:
        m_curFog = static_cast<uint8_t>((value >> 56) & 0xFF);
        break;
    case GS_REG_TEX1_1:
    case GS_REG_TEX1_2:
    {
        int ci = (regAddr == GS_REG_TEX1_2) ? 1 : 0;
        m_ctx[ci].tex1 = value;
        break;
    }
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    {
        int ci = (regAddr == GS_REG_TEX2_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        m_backend->LoadClut(t, m_texclut);
        break;
    }
    case GS_REG_XYOFFSET_1:
    case GS_REG_XYOFFSET_2:
    {
        int ci = (regAddr == GS_REG_XYOFFSET_2) ? 1 : 0;
        if (traceGsViewportEnabled())
        {
            const uint64_t oldValue = static_cast<uint64_t>(m_ctx[ci].xyoffset.ofx) |
                                      (static_cast<uint64_t>(m_ctx[ci].xyoffset.ofy) << 32);
            traceGsViewportWrite(ci ? "XYOFFSET_2" : "XYOFFSET_1", oldValue, value & 0x0000FFFF0000FFFFull, true);
        }
        m_ctx[ci].xyoffset.ofx = static_cast<uint16_t>(value & 0xFFFF);
        m_ctx[ci].xyoffset.ofy = static_cast<uint16_t>((value >> 32) & 0xFFFF);
        break;
    }
    case GS_REG_PRMODECONT:
    {
        m_prmodecont = (value & 1) != 0;
        const GSPrimType type = m_primRegister.type;
        m_prim = m_prmodecont ? m_primRegister : m_prmodeRegister;
        m_prim.type = type;
        break;
    }
    case GS_REG_PRMODE:
    {
        m_prmodeRegister = decodePrimRegister(value);
        if (!m_prmodecont)
        {
            const GSPrimType type = m_primRegister.type;
            m_prim = m_prmodeRegister;
            m_prim.type = type;
        }
        break;
    }
    case GS_REG_TEXCLUT:
        m_texclut.cbw = static_cast<uint8_t>(value & 0x3Fu);
        m_texclut.cou = static_cast<uint8_t>((value >> 6) & 0x3Fu);
        m_texclut.cov = static_cast<uint16_t>((value >> 12) & 0x3FFu);
        break;
    case GS_REG_SCISSOR_1:
    case GS_REG_SCISSOR_2:
    {
        int ci = (regAddr == GS_REG_SCISSOR_2) ? 1 : 0;
        if (traceGsViewportEnabled())
        {
            const uint64_t oldValue = static_cast<uint64_t>(m_ctx[ci].scissor.x0) |
                                      (static_cast<uint64_t>(m_ctx[ci].scissor.x1) << 16) |
                                      (static_cast<uint64_t>(m_ctx[ci].scissor.y0) << 32) |
                                      (static_cast<uint64_t>(m_ctx[ci].scissor.y1) << 48);
            traceGsViewportWrite(ci ? "SCISSOR_2" : "SCISSOR_1", oldValue, value & 0x07FF07FF07FF07FFull, false);
        }
        m_ctx[ci].scissor.x0 = static_cast<uint16_t>(value & 0x7FF);
        m_ctx[ci].scissor.x1 = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_ctx[ci].scissor.y0 = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_ctx[ci].scissor.y1 = static_cast<uint16_t>((value >> 48) & 0x7FF);
        break;
    }
    case GS_REG_ALPHA_1:
    case GS_REG_ALPHA_2:
    {
        int ci = (regAddr == GS_REG_ALPHA_2) ? 1 : 0;
        m_ctx[ci].alpha = value;
        break;
    }
    case GS_REG_TEST_1:
    case GS_REG_TEST_2:
    {
        int ci = (regAddr == GS_REG_TEST_2) ? 1 : 0;
        m_ctx[ci].test = value;
        break;
    }
    case GS_REG_FRAME_1:
    case GS_REG_FRAME_2:
    {
        int ci = (regAddr == GS_REG_FRAME_2) ? 1 : 0;
        m_ctx[ci].frame.fbp = static_cast<uint32_t>(value & 0x1FF);
        uint32_t fbw = static_cast<uint32_t>((value >> 16) & 0x3F);
        if (fbw == 0u)
        {
            fbw = 10u; // Default to 10 blocks (640 / 64) when FBW is 0
        }
        m_ctx[ci].frame.fbw = fbw;
        m_ctx[ci].frame.psm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_ctx[ci].frame.fbmsk = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        {
            static uint32_t s_frameLogs = 0u;
            if (s_frameLogs < 32u)
            {
                std::fprintf(stderr,
                             "[gs:frame] ctx=%d fbp=%u fbw=%u psm=0x%x fbmsk=0x%x\n",
                             ci,
                             m_ctx[ci].frame.fbp,
                             m_ctx[ci].frame.fbw,
                             static_cast<uint32_t>(m_ctx[ci].frame.psm),
                             m_ctx[ci].frame.fbmsk);
                ++s_frameLogs;
            }
        }
        break;
    }
    case GS_REG_ZBUF_1:
    case GS_REG_ZBUF_2:
    {
        int ci = (regAddr == GS_REG_ZBUF_2) ? 1 : 0;
        m_ctx[ci].zbuf.zbp = value & 0x1FF;
        m_ctx[ci].zbuf.psm = ((value >> 24) & 0xF) | 0x30;
        m_ctx[ci].zbuf.zmask = (value >> 32) & 1;
        break;
    }
    case GS_REG_FBA_1:
    case GS_REG_FBA_2:
    {
        int ci = (regAddr == GS_REG_FBA_2) ? 1 : 0;
        m_ctx[ci].fba = value;
        break;
    }
    case GS_REG_BITBLTBUF:
    {
        m_bitbltbuf.sbp = static_cast<uint32_t>(value & 0x3FFF);
        m_bitbltbuf.sbw = static_cast<uint8_t>((value >> 16) & 0x3F);
        m_bitbltbuf.spsm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_bitbltbuf.dbp = static_cast<uint32_t>((value >> 32) & 0x3FFF);
        m_bitbltbuf.dbw = static_cast<uint8_t>((value >> 48) & 0x3F);
        m_bitbltbuf.dpsm = static_cast<uint8_t>((value >> 56) & 0x3F);
        break;
    }
    case GS_REG_TRXPOS:
    {
        m_trxpos.ssax = static_cast<uint16_t>(value & 0x7FF);
        m_trxpos.ssay = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_trxpos.dsax = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_trxpos.dsay = static_cast<uint16_t>((value >> 48) & 0x7FF);
        m_trxpos.dir = static_cast<uint8_t>((value >> 59) & 0x3);
        break;
    }
    case GS_REG_TRXREG:
    {
        m_trxreg.rrw = static_cast<uint16_t>(value & 0xFFF);
        m_trxreg.rrh = static_cast<uint16_t>((value >> 32) & 0xFFF);
        break;
    }
    case GS_REG_TRXDIR:
    {
        m_trxdir = static_cast<uint32_t>(value & 0x3);

        if (m_backend)
        {
            GSTransferCommand command{};
            command.bitbltbuf = m_bitbltbuf;
            command.trxpos = m_trxpos;
            command.trxreg = m_trxreg;
            command.direction = m_trxdir;
            m_backend->BeginTransfer(command);
        }
        if (m_trxdir == 0u && m_bitbltbuf.dbp >= 0x3400u && m_bitbltbuf.dbp < 0x3E00u &&
            s_texPoolLines < 800u && texPoolTraceActive())
        {
            std::snprintf(s_texPoolUploads[s_texPoolUploadSeq & 7u], sizeof(s_texPoolUploads[0]),
                          "[texpool] upload #%llu dbp=0x%x dbw=%u dpsm=0x%x rr=%ux%u dsa=(%u,%u)\n",
                          static_cast<unsigned long long>(s_texPoolUploadSeq), m_bitbltbuf.dbp,
                          static_cast<uint32_t>(m_bitbltbuf.dbw), static_cast<uint32_t>(m_bitbltbuf.dpsm),
                          static_cast<uint32_t>(m_trxreg.rrw), static_cast<uint32_t>(m_trxreg.rrh),
                          static_cast<uint32_t>(m_trxpos.dsax), static_cast<uint32_t>(m_trxpos.dsay));
            ++s_texPoolUploadSeq;
            if (s_texPoolUploadCount < 8u)
                ++s_texPoolUploadCount;
            ++s_texPoolUploadsSinceArrow;
        }
        {
            static uint32_t s_blits = 0u;
            static uint32_t s_blitTotal = 0u;
            static uint32_t s_uniqueHigh = 0u;
            static uint32_t s_fbBlits = 0u;
            ++s_blitTotal;
            const bool highTex = (m_bitbltbuf.dpsm == 0x2u &&
                                  m_bitbltbuf.dbp >= 0x3400u && m_bitbltbuf.dbp < 0x3C00u);
            const bool fbBlit = (m_bitbltbuf.dbp < 0x2000u && m_trxdir == 0u);
            if ((m_bitbltbuf.dbp >= 0x2000u && s_blits < 16u) ||
                (highTex && s_uniqueHigh < 128u) ||
                (fbBlit && s_fbBlits < 16u))
            {
                std::fprintf(stderr,
                             "[gs:blit] n=%u total=%u dir=%u dbp=0x%x dbw=%u dpsm=0x%x sbp=0x%x "
                             "rrw=%u rrh=%u dsax=%u dsay=%u\n",
                             s_blits,
                             s_blitTotal,
                             m_trxdir,
                             m_bitbltbuf.dbp,
                             static_cast<uint32_t>(m_bitbltbuf.dbw),
                             static_cast<uint32_t>(m_bitbltbuf.dpsm),
                             m_bitbltbuf.sbp,
                             static_cast<uint32_t>(m_trxreg.rrw),
                             static_cast<uint32_t>(m_trxreg.rrh),
                             static_cast<uint32_t>(m_trxpos.dsax),
                             static_cast<uint32_t>(m_trxpos.dsay));
                if (m_bitbltbuf.dbp >= 0x2000u && s_blits < 16u)
                    ++s_blits;
                if (highTex)
                    ++s_uniqueHigh;
                if (fbBlit)
                    ++s_fbBlits;
            }
        }
        recordTransferDebugEventUnlocked();
        break;
    }
    case GS_REG_HWREG:
    {
        uint8_t buf[8];
        std::memcpy(buf, &value, 8);
        processImageData(buf, 8);
        break;
    }
    case GS_REG_PABE:
        m_pabe = (value & 1u) != 0u;
        break;
    case GS_REG_FOGCOL:
        m_fogR = static_cast<uint8_t>(value & 0xFFu);
        m_fogG = static_cast<uint8_t>((value >> 8) & 0xFFu);
        m_fogB = static_cast<uint8_t>((value >> 16) & 0xFFu);
        break;
    case GS_REG_TEXFLUSH:
        if (m_backend)
            m_backend->TextureFlush();
        break;
    case GS_REG_SCANMSK:
        m_scanmsk = value;
        break;
    case GS_REG_DIMX:
        m_dimx = value;
        break;
    case GS_REG_DTHE:
        m_dthe = value;
        break;
    case GS_REG_COLCLAMP:
        m_colclamp = value;
        break;
    case GS_REG_MIPTBP1_1:
    case GS_REG_MIPTBP1_2:
    {
        const int ci = (regAddr == GS_REG_MIPTBP1_2) ? 1 : 0;
        m_ctx[ci].miptbp1 = value;
        break;
    }
    case GS_REG_MIPTBP2_1:
    case GS_REG_MIPTBP2_2:
    {
        const int ci = (regAddr == GS_REG_MIPTBP2_2) ? 1 : 0;
        m_ctx[ci].miptbp2 = value;
        break;
    }
    case GS_REG_TEXA:
    {
        m_texa.ta0 = static_cast<uint8_t>(value & 0xFFu);
        m_texa.aem = ((value >> 15) & 0x1u) != 0u;
        m_texa.ta1 = static_cast<uint8_t>((value >> 32) & 0xFFu);
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t texaIndex = s_debugTexaWriteCount.fetch_add(1u, std::memory_order_relaxed);
            if (texaIndex < 24u)
            {
                RUNTIME_LOG("[gs:texa] idx=" << texaIndex
                                             << " value=0x" << std::hex << value
                                             << " ta0=0x" << ((value >> 0) & 0xFFu)
                                             << " aem=" << ((value >> 15) & 0x1u)
                                             << " ta1=0x" << ((value >> 32) & 0xFFu)
                                             << std::dec
                                             << std::endl);
            }
        });
        break;
    }
    case GS_REG_SIGNAL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t lo = static_cast<uint32_t>(m_privRegs->siglblid & 0xFFFFFFFF);
            lo = (lo & ~mask) | (id & mask);
            m_privRegs->siglblid = (m_privRegs->siglblid & 0xFFFFFFFF00000000ULL) | lo;
            m_privRegs->csr.fetch_or(0x1);
        }
        break;
    }
    case GS_REG_FINISH:
    {
        if (m_backend)
        {
            m_backend->Flush();
            m_backend->Sync(GSSyncReason::Finish);
        }
        if (m_privRegs)
            m_privRegs->csr.fetch_or(0x2);
        break;
    }
    case GS_REG_LABEL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t hi = static_cast<uint32_t>(m_privRegs->siglblid >> 32);
            hi = (hi & ~mask) | (id & mask);
            m_privRegs->siglblid = (static_cast<uint64_t>(hi) << 32) | (m_privRegs->siglblid & 0xFFFFFFFF);
        }
        break;
    }
    case 0x59:
        if (m_privRegs)
            m_privRegs->dispfb1 = value;
        break;
    case 0x5a:
        if (m_privRegs)
        {
            // Repair sign-extended DISPLAY (DW=0xFFF / DH=0x7FF) from GIF PATH3.
            const uint32_t dw = static_cast<uint32_t>((value >> 32) & 0x0FFFu);
            const uint32_t dh = static_cast<uint32_t>((value >> 44) & 0x07FFu);
            if (dw == 0x0FFFu || dh == 0x07FFu)
            {
                value = (652ull << 0) | (50ull << 12) | (3ull << 23) | (0ull << 27) |
                        (2556ull << 32) | (447ull << 44);
            }
            m_privRegs->display1 = value;
        }
        break;
    case 0x5b:
        if (m_privRegs)
            m_privRegs->dispfb2 = value;
        break;
    case 0x5c:
        if (m_privRegs)
        {
            const uint32_t dw = static_cast<uint32_t>((value >> 32) & 0x0FFFu);
            const uint32_t dh = static_cast<uint32_t>((value >> 44) & 0x07FFu);
            if (dw == 0x0FFFu || dh == 0x07FFu)
            {
                value = (652ull << 0) | (50ull << 12) | (3ull << 23) | (0ull << 27) |
                        (2556ull << 32) | (447ull << 44);
            }
            m_privRegs->display2 = value;
        }
        break;
    case 0x5f:
        if (m_privRegs)
            m_privRegs->bgcolor = value;
        break;
    default:
        break;
    }

    recordRegisterDebugEventUnlocked(regAddr, value);
}

void GS::vertexKick(bool drawing)
{
    ++m_vtxCount;
    ++m_vtxIndex;

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t debugIndex = s_debugGsVertexKickCount.fetch_add(1, std::memory_order_relaxed);
        if (debugIndex < 96u)
        {
            RUNTIME_LOG("[gs:kick] idx=" << debugIndex
                                         << " drawing=" << static_cast<uint32_t>(drawing ? 1u : 0u)
                                         << " prim=" << static_cast<uint32_t>(m_prim.type)
                                         << " vtxCount=" << m_vtxCount
                                         << std::endl);
        }
    });

    int needed = 0;
    switch (m_prim.type)
    {
    case GS_PRIM_POINT:
        needed = 1;
        break;
    case GS_PRIM_LINE:
        needed = 2;
        break;
    case GS_PRIM_LINESTRIP:
        needed = 2;
        break;
    case GS_PRIM_TRIANGLE:
        needed = 3;
        break;
    case GS_PRIM_TRISTRIP:
        needed = 3;
        break;
    case GS_PRIM_TRIFAN:
        needed = 3;
        break;
    case GS_PRIM_SPRITE:
        needed = 2;
        break;
    default:
        return;
    }

    if (m_vtxCount < needed)
        return;

    if (drawing && m_backend)
    {
        GSPrimitiveBatch batch = buildDrawBatch(needed);
        if (batch.state.prim.tme && batch.state.context.tex0.tw == 5u && batch.state.context.tex0.th == 6u &&
            s_texPoolLines < 800u && texPoolTraceActive())
        {
            const GSTex0Reg &tex0 = batch.state.context.tex0;
            if (tex0.tbp0 != s_texPoolLastArrowTbp || s_texPoolUploadsSinceArrow != 0u)
            {
                if (s_texPoolArrowRun > 1u)
                    std::fprintf(stderr, "[texpool]   (+%u more arrow prims)\n", s_texPoolArrowRun - 1u);
                const uint32_t shown = std::min(s_texPoolUploadsSinceArrow, s_texPoolUploadCount);
                for (uint32_t i = shown; i > 0u; --i)
                {
                    ++s_texPoolLines;
                    std::fputs(s_texPoolUploads[(s_texPoolUploadSeq - i) & 7u], stderr);
                }
                s_texPoolUploadsSinceArrow = 0u;
                ++s_texPoolLines;
                std::fprintf(stderr,
                             "[texpool] draw tbp=0x%x tbw=%u psm=0x%x tcc=%u tfx=%u cbp=0x%x cpsm=0x%x fbp=%u "
                             "type=%u xy=(%.1f,%.1f) q=%g rgba=(%u,%u,%u,%u)\n",
                             tex0.tbp0, static_cast<uint32_t>(tex0.tbw), static_cast<uint32_t>(tex0.psm),
                             static_cast<uint32_t>(tex0.tcc), static_cast<uint32_t>(tex0.tfx), tex0.cbp,
                             static_cast<uint32_t>(tex0.cpsm), batch.state.context.frame.fbp,
                             static_cast<uint32_t>(batch.state.prim.type), batch.vertices[0].x,
                             batch.vertices[0].y, batch.vertices[0].q,
                             static_cast<uint32_t>(batch.vertices[0].r), static_cast<uint32_t>(batch.vertices[0].g),
                             static_cast<uint32_t>(batch.vertices[0].b), static_cast<uint32_t>(batch.vertices[0].a));
                s_texPoolLastArrowTbp = tex0.tbp0;
                s_texPoolArrowRun = 1u;
            }
            else
            {
                ++s_texPoolArrowRun;
            }
        }
        updatePreferredDisplaySourceForDraw(batch);
        // Triangles that would draw no pixel skip the copy, queue and band
        // work (about half of race triangles are sub-pixel). ATV_GS_NO_CULL=1 keeps them.
        static const bool s_noCull = std::getenv("ATV_GS_NO_CULL") != nullptr;
        const bool triangle = batch.state.prim.type == GS_PRIM_TRIANGLE ||
                              batch.state.prim.type == GS_PRIM_TRISTRIP || batch.state.prim.type == GS_PRIM_TRIFAN;
        if (s_noCull || !triangle || !gsTriangleDrawsNothing(batch))
            m_backend->Submit(batch);
        recordDrawDebugEventUnlocked(needed);
    }

    switch (m_prim.type)
    {
    case GS_PRIM_LINE:
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_SPRITE:
    case GS_PRIM_POINT:
        m_vtxCount = 0;
        break;
    case GS_PRIM_LINESTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxCount = 1;
        break;
    case GS_PRIM_TRISTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxCount = 2;
        break;
    case GS_PRIM_TRIFAN:
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxCount = 2;
        break;
    default:
        m_vtxCount = 0;
        break;
    }
}

void GS::processImageData(const uint8_t *data, uint32_t sizeBytes)
{
    if (m_backend)
        m_backend->UploadImage(data, sizeBytes);
}


bool GS::clearFramebufferContext(uint32_t contextIndex, uint32_t rgba)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend && m_backend->ClearFramebuffer(m_ctx[(contextIndex != 0u) ? 1 : 0], rgba);
}

bool GS::clearActiveFramebuffer(uint32_t rgba)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend && m_backend->ClearFramebuffer(activeContext(), rgba);
}

uint32_t GS::consumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend ? m_backend->ConsumeLocalToHostBytes(dst, maxBytes) : 0u;
}

void GS::setRasterBackend(std::unique_ptr<GSRasterBackend> backend)
{
    GSApiScope apiScope{__func__};
    if (!backend)
        backend = makeDefaultRasterBackend();

    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    if (m_backend)
    {
        m_backend->Flush();
        m_backend->Sync(GSSyncReason::Reset);

        // The external 4 MiB GS allocation is the backend hand-off format.
        // This keeps hot backend replacement deterministic even when a future
        // GPU backend keeps a private/mirrored local-memory representation.
        if (m_localMemoryStorage && m_localMemorySize != 0u)
        {
            std::vector<uint8_t> localMemory;
            m_backend->SnapshotVram(localMemory);
            const size_t bytes = std::min<size_t>(localMemory.size(), m_localMemorySize);
            if (bytes != 0u)
                std::memcpy(m_localMemoryStorage, localMemory.data(), bytes);
        }
    }

    m_backend = std::move(backend);
    m_backend->Initialize(m_localMemoryStorage, m_localMemorySize);
}

uint32_t GS::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend ? m_backend->ReadVram(psm, base, bw, x, y) : 0u;
}

void GS::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    GSApiScope apiScope{__func__};
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (m_backend)
        m_backend->WriteVram(psm, base, bw, x, y, value);
}

GSPrimitiveBatch GS::buildDrawBatch(int vertexCount) const
{
    GSPrimitiveBatch batch{};
    batch.vertexCount = static_cast<uint8_t>(std::min(vertexCount, 3));
    for (int i = 0; i < batch.vertexCount; ++i)
        batch.vertices[static_cast<size_t>(i)] = m_vtxQueue[i];
    batch.state.context = m_ctx[m_prim.ctxt ? 1 : 0];
    batch.state.prim = m_prim;
    batch.state.texa = m_texa;
    batch.state.texclut = m_texclut;
    batch.state.pabe = m_pabe;
    batch.state.scanmsk = m_scanmsk;
    batch.state.dimx = m_dimx;
    batch.state.dthe = m_dthe;
    batch.state.colclamp = m_colclamp;
    batch.state.fogR = m_fogR;
    batch.state.fogG = m_fogG;
    batch.state.fogB = m_fogB;
    batch.state.textureWidth = static_cast<uint16_t>(1u << std::min<uint32_t>(batch.state.context.tex0.tw, 10u));
    batch.state.textureHeight = static_cast<uint16_t>(1u << std::min<uint32_t>(batch.state.context.tex0.th, 10u));
    const uint64_t tex1 = batch.state.context.tex1;
    const uint8_t mmag = static_cast<uint8_t>((tex1 >> 5u) & 0x1u);
    const uint8_t mmin = static_cast<uint8_t>((tex1 >> 6u) & 0x7u);
    batch.state.linearFilter = mmag != 0u || mmin == 1u || (mmin & 0x4u) != 0u;
    return batch;
}

void GS::updatePreferredDisplaySourceForDraw(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSContext &ctx = state.context;
    // ATV_TRACE_FLIP=1 (diagnostic): draws into the buffer DISPFB1 currently
    // shows vs. into other buffers, printed every 20000 draws.
    static const bool s_traceFlip = std::getenv("ATV_TRACE_FLIP") != nullptr;
    if (s_traceFlip && m_privRegs)
    {
        static uint64_t s_front = 0, s_back = 0, s_frontRuns = 0;
        static bool s_lastFront = false;
        const uint32_t shown = static_cast<uint32_t>(m_privRegs->dispfb1 & 0x1FFu);
        const bool front = ctx.frame.fbp == shown;
        if (front)
        {
            ++s_front;
            if (!s_lastFront)
                ++s_frontRuns;
        }
        else
            ++s_back;
        s_lastFront = front;
        if ((s_front + s_back) % 20000u == 0u)
            std::fprintf(stderr, "[flip-trace] draws front=%llu back=%llu frontRuns=%llu shownFbp=%u drawFbp=%u\n",
                         static_cast<unsigned long long>(s_front), static_cast<unsigned long long>(s_back),
                         static_cast<unsigned long long>(s_frontRuns), shown, ctx.frame.fbp);
    }
    if (m_hasPreferredDisplaySource && ctx.frame.fbp == m_preferredDisplayDestFbp)
        m_hasPreferredDisplaySource = false;
    if (state.prim.type != GS_PRIM_SPRITE || batch.vertexCount < 2u)
        return;

    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    int x0 = static_cast<int>(v0.x) - (ctx.xyoffset.ofx >> 4);
    int y0 = static_cast<int>(v0.y) - (ctx.xyoffset.ofy >> 4);
    int x1 = static_cast<int>(v1.x) - (ctx.xyoffset.ofx >> 4);
    int y1 = static_cast<int>(v1.y) - (ctx.xyoffset.ofy >> 4);
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    const int xEnd = x0 + std::max(1, x1 - x0) - 1;
    const int yEnd = y0 + std::max(1, y1 - y0) - 1;
    const uint8_t alphaMode = static_cast<uint8_t>(ctx.alpha & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((ctx.alpha >> 32u) & 0xFFu);
    const bool displayCopy = state.prim.tme && state.prim.abe && state.prim.fst && state.prim.ctxt &&
                             ctx.frame.fbp != ctx.tex0.tbp0 && alphaMode == 0x64u &&
                             (alphaFix == 0x60u || alphaFix == 0x80u) &&
                             x0 <= 0 && y0 <= 0 && xEnd >= 639 && yEnd >= 447;
    if (displayCopy)
    {
        static uint32_t s_prefLogs = 0u;
        if (s_prefLogs < 4u)
        {
            std::fprintf(stderr,
                         "[gs:pref] destFbp=%u tbp0=%u tbw=%u psm=0x%x alpha=0x%x fix=0x%x\n",
                         ctx.frame.fbp,
                         ctx.tex0.tbp0,
                         static_cast<uint32_t>(ctx.tex0.tbw),
                         static_cast<uint32_t>(ctx.tex0.psm),
                         alphaMode,
                         alphaFix);
            ++s_prefLogs;
        }
        m_preferredDisplaySourceFrame = {ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0u};
        m_preferredDisplayDestFbp = ctx.frame.fbp;
        m_hasPreferredDisplaySource = true;
    }
}
