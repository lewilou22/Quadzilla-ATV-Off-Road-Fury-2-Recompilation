#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// True when GSCpuBackend::DrawTriangle would write no pixel for this
// triangle (rejected, outside the scissor, or a small triangle whose sample
// points all miss). Exact for the boxes it tests; larger triangles return false.
bool gsTriangleDrawsNothing(const GSPrimitiveBatch &batch);

// Record tags of an ATV_GS_CAPTURE file (see gs_cpu_backend.cpp).
enum GSCaptureTag : uint8_t
{
    GSCaptureSubmit = 1,
    GSCaptureLoadClut,
    GSCaptureBeginTransfer,
    GSCaptureUploadImage,
    GSCaptureFlush,
    GSCaptureTextureFlush,
    GSCaptureSync,
    GSCaptureClear,
    GSCaptureWriteVram,
    GSCaptureReadVram,
    GSCaptureConsume,
    GSCaptureSnapshot,
    GSCaptureEnd = 0xFF,
};

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    // Submit for a run of primitives under one lock (the threaded backend's
    // worker hands over consecutive primitive commands this way).
    void SubmitMany(const GSPrimitiveBatch *const *batches, size_t count);
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    // Display conversion of a local-memory snapshot (SnapshotVram output),
    // without any backend lock. Present() = SnapshotVram + this.
    static PresentationFrame PresentSnapshot(std::vector<uint8_t> &snapshot, const GSPresentationRequest &request);

    // Replay support: the palette state a capture file starts with.
    void RestoreClutState(const std::array<uint16_t, 512> &clut, const std::array<uint32_t, 2> &cbp);

private:
    void MaybeStartCapture();
    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{};
    std::array<uint32_t, 2> m_clutCbp{};
    GSMem::TexturePageCache m_texturePageCache;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;

    // Optional banded rasterization (ATV_GS_BANDS=N): primitives are batched
    // and every band thread draws the whole batch restricted to its own rows.
    struct BandWrite
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        uint32_t layoutKey = 0;
    };
    void StartBandThreads();
    void QueueBandedPrimitive(const GSPrimitiveBatch &batch);
    void SubmitUnlocked(const GSPrimitiveBatch &batch);
    enum class BandFlushReason : uint8_t
    {
        Other, Alias, Batch, Hazard, Clut, Texture, Sync, Read, Write, Upload, Clear, Count
    };
    void FlushBands(BandFlushReason reason = BandFlushReason::Other) const;
    void WaitBandsIdle() const;
    void BandWorker(uint32_t band);
    void DrawBand(uint32_t band);

    uint32_t m_bandCount = 1;
    mutable std::vector<GSPrimitiveBatch> m_bandPending;
    // TEXFLUSH boundaries are replayed by each band in command order.
    mutable std::vector<size_t> m_bandTextureFlushes;
    mutable std::vector<BandWrite> m_bandWrites;
    // Whole pages the queued draws may sample. A host-to-local upload whose
    // destination misses these and m_bandWrites runs without flushing the
    // queue (ATV_GS_UPLOAD_FLUSH=1 always flushes).
    mutable std::vector<BandWrite> m_bandReads;
    // m_bandWrites/m_bandReads of the batch the band threads are drawing: an
    // upload outside them need not wait for that batch (InFlightTouches).
    mutable std::vector<BandWrite> m_bandInFlightWrites;
    mutable std::vector<BandWrite> m_bandInFlightReads;
    bool InFlightTouches(const BandWrite &range) const;
    // CLUT seen by queued band primitives: the CLUT when the first of them was queued plus each
    // later load (pending index, CLUT). Bands draw from these snapshots, so a palette load does not
    // flush the queue (ATV2 loads ~1000 palettes a frame). LoadClut still flushes when queued
    // draws write the palette's source or a pending TEXFLUSH leaves the shared page cache stale.
    using ClutTable = std::array<uint16_t, 512>;
    mutable ClutTable m_bandPendingClutStart{};
    mutable std::vector<std::pair<size_t, ClutTable>> m_bandClutChanges;
    mutable ClutTable m_bandInFlightClutStart{};
    mutable std::vector<std::pair<size_t, ClutTable>> m_bandInFlightClutChanges;
    bool PendingWritesTouch(const BandWrite &range) const;
    bool PendingReadsTouch(const BandWrite &range) const;
    // The upload must flush queued draws: they write its pages, or freezing
    // is off and they read them, or its range is unbounded or too large.
    bool UploadNeedsFlush() const;
    // An upload over pages queued draws only read keeps the old page bytes
    // for those draws instead of flushing them (FreezePendingPages): span
    // [from, to) of the pending batch's primitives reads `buffer`.
    struct FrozenSpan
    {
        uint32_t pageBase, from, to, buffer;
    };
    void FreezePendingPages(const BandWrite &range);
    bool m_freezeUploads = true;
    mutable std::vector<FrozenSpan> m_bandFrozen;
    mutable std::vector<std::vector<uint8_t>> m_bandFrozenBytes;
    mutable std::vector<std::vector<uint8_t>> m_bandInFlightFrozenBytes;
    mutable std::vector<std::vector<uint8_t>> m_frozenPool;
    mutable std::vector<std::pair<uint32_t, uint32_t>> m_pageVersionStart; // page -> first primitive of its contents
    // Per texture-flush segment of the in-flight batch: frozen pages it reads.
    mutable std::vector<std::vector<GSMem::FrozenPage>> m_bandInFlightSegments;
    // Everything QueueBandedPrimitive's ranges depend on. A primitive with the
    // key of the last one queued since the last flush adds no ranges and hits
    // no alias or hazard (that one passed the same checks), so it is appended.
    struct BandKey
    {
        uint64_t clamp = 0;
        uint32_t fbp = 0, fbw = 0, fpsm = 0, rows = 0, zbp = 0, zpsm = 0, tbp0 = 0, tbw = 0, tpsm = 0;
        uint32_t texW = 0, texH = 0, flags = 0;
        bool operator==(const BandKey &) const = default;
    };
    mutable BandKey m_lastBandKey{};
    mutable bool m_lastBandKeyValid = false;
    bool PendingTouches(const BandWrite &range) const;
    BandWrite m_uploadRange{};
    bool m_uploadRangeValid = false;
    bool m_uploadNoFlush = true;
    // The batch the band threads are drawing. With ATV_GS_BAND_PIPELINE every
    // band has its own thread and a full batch is drawn while the next one is
    // queued; anything else that touches VRAM, the CLUT or the texture caches
    // waits for it first (WaitBandsIdle).
    mutable std::vector<GSPrimitiveBatch> m_bandInFlight;
    mutable std::vector<size_t> m_bandInFlightTextureFlushes;
    mutable bool m_bandInFlightActive = false;
    bool m_bandPipeline = false;
    std::vector<std::unique_ptr<GSMem::TexturePageCache>> m_bandCaches;
    std::vector<std::thread> m_bandThreads;
    mutable std::mutex m_bandMutex;
    mutable std::condition_variable m_bandStart;
    mutable std::condition_variable m_bandDone;
    mutable uint64_t m_bandGeneration = 0;
    mutable uint32_t m_bandRemaining = 0;
    bool m_bandStop = false;
    bool m_legacyTextureFlush = false;
    bool m_profileBands = false;
    // ATV_GS_BATCH_SIZE; 4096 cuts band wake/done handoffs (~1.5 fps in races vs 1024).
    size_t m_bandBatchLimit = 4096;
    mutable std::array<uint64_t, 16> m_bandDrawNs{};
    mutable std::array<uint64_t, 16> m_bandAccumulatedNs{};
    mutable std::array<uint64_t, static_cast<size_t>(BandFlushReason::Count)> m_bandFlushCounts{};
    mutable uint64_t m_bandProfileStartNs = 0;
    mutable uint64_t m_bandFlushNs = 0;
    mutable uint64_t m_bandWaitNs = 0;
    mutable uint64_t m_bandPrimitiveCount = 0;
};
