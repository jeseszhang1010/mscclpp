// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_

#include <cstdint>

#include "assert_device.hpp"
#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE) && defined(MSCCLPP_USE_GPUNETIO)
#include <doca_gpunetio_device.h>

#include "bulk_device.hpp"

// A run-time-selected GPUNetIO submission path, kept out of line (see GpuNetIoDeviceHandle).
#define MSCCLPP_GPUNETIO_PATH __noinline__ __device__
#endif

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE) && defined(MSCCLPP_USE_GPUNETIO)
namespace detail {
// Non-leader lanes share per-warp BF scratch; these locks protect it across divergent calls.
#if defined(__CUDACC_RDC__)
inline
#else
static
#endif
    __device__ unsigned int gpuNetIoWqeLocks[1024] = {};
}  // namespace detail
#endif

/// A registered memory region that a GPUNetIO PortChannel can address by MemoryId.
struct GpuNetIoMemoryDeviceHandle {
  /// Base address of the region in the owner's address space.
  uint64_t addr;
  /// Size of the region in bytes. Zero marks a MemoryId that is not usable on this channel.
  uint64_t size;
  /// Local key (local regions) or remote key (remote regions), in network byte order.
  uint32_t key;
  /// Nonzero if the region belongs to the local process.
  uint32_t isLocal;
};

/// GPUNetIO submission features selected at queue-pair creation (GpuNetIoDeviceHandle::features).
enum GpuNetIoFeature : uint32_t {
  /// BlueFlame submission of WQEs prepared in shared memory (`MSCCLPP_GPUNETIO_BLUEFLAME`).
  GpuNetIoFeatureBlueFlame = 1u << 0,
  /// Exclusive QPs, inline signals and payloads, direct atomics, no system fences (`MSCCLPP_GPUNETIO_OPT`).
  GpuNetIoFeatureOpt = 1u << 1,
};

/// Device-side state of a PortChannel whose connection uses EndpointConfig::Ib::Mode::GpuNetIo.
/// The calling GPU thread builds work requests and rings the NIC doorbell itself (DOCA GPUNetIO).
/// Callers must make source data ready before put operations; DOCA orders queue publication.
/// All operations on a QP must be serialized, including across channels and submitting threads.
struct GpuNetIoDeviceHandle {
  /// GPU-resident DOCA queue pair (`doca_gpu_dev_verbs_qp*`) of the connection.
  void* qp;
  /// Remote semaphore token targeted by signals.
  uint64_t signalAddr;
  /// Remote key of the semaphore token, in network byte order.
  uint32_t signalKey;
  /// Local key of the connection's atomic fetch-result scratch, in network byte order.
  uint32_t atomicScratchKey;
  /// Connection-owned 8-byte buffer that receives atomic fetch results.
  uint64_t atomicScratchAddr;
  /// Device-resident outbound semaphore counter for inline signal writes.
  uint64_t* outboundToken;
  /// Snapshot of the ProxyService memory table, indexed by MemoryId.
  const GpuNetIoMemoryDeviceHandle* memories;
  /// Number of entries in `memories`.
  uint32_t numMemories;
  /// GpuNetIoFeature flags of the queue pair.
  uint32_t features;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE static void trap() {
#if defined(MSCCLPP_DEVICE_CUDA)
    __trap();
#else
    __builtin_trap();
#endif
  }

  /// Trap on an invalid access even when device assertions are disabled.
  MSCCLPP_DEVICE_INLINE static void require(bool valid) {
    if (!valid) {
      MSCCLPP_ASSERT_DEVICE(false, "Invalid GPUNetIO PortChannel memory access");
      trap();
    }
  }

  /// Resolve a MemoryId and check its owner and bounds.
  MSCCLPP_DEVICE_INLINE GpuNetIoMemoryDeviceHandle memory(uint32_t id, bool local, uint64_t offset,
                                                          uint64_t bytes) const {
    require(id < numMemories);
    const GpuNetIoMemoryDeviceHandle mem = memories[id];
    require((mem.isLocal != 0) == local && offset <= mem.size && bytes <= mem.size - offset);
    return mem;
  }

  /// True if work requests are submitted through BlueFlame.
  MSCCLPP_DEVICE_INLINE bool blueFlame() const { return (features & GpuNetIoFeatureBlueFlame) != 0; }

  /// True if the exclusive-QP / inline-signal optimizations are enabled.
  MSCCLPP_DEVICE_INLINE bool optimized() const { return (features & GpuNetIoFeatureOpt) != 0; }

  /// RDMA write of `size` bytes from a local region to a remote region.
  MSCCLPP_DEVICE_INLINE void put(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                 const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (optimized()) {
      if (blueFlame()) {
        putImpl(dst, dstOffset, src, srcOffset, size);
      } else {
        putDb(dst, dstOffset, src, srcOffset, size);
      }
    } else {
      if (blueFlame()) {
        putSharedBf(dst, dstOffset, src, srcOffset, size);
      } else {
        putBaseline(dst, dstOffset, src, srcOffset, size);
      }
    }
#else
    trap();
#endif
  }

  /// Signal the remote semaphore: an inline token write when optimized, otherwise an RDMA atomic +1.
  MSCCLPP_DEVICE_INLINE void signal() {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (!optimized()) {
      atomicAdd(signalAddr, signalKey, 1);
    } else if (blueFlame()) {
      signalBf();
    } else {
      signalDb();
    }
#else
    trap();
#endif
  }

  /// RDMA write followed by a semaphore signal, ordered after the payload write.
  MSCCLPP_DEVICE_INLINE void putWithSignal(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                           const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (optimized()) {
      if (blueFlame()) {
        putWithSignalBf(dst, dstOffset, src, srcOffset, size);
      } else {
        putWithSignalDb(dst, dstOffset, src, srcOffset, size);
      }
    } else {
      if (blueFlame()) {
        putWithSignalSharedBf(dst, dstOffset, src, srcOffset, size);
      } else {
        putWithSignalBaseline(dst, dstOffset, src, srcOffset, size);
      }
    }
#else
    trap();
#endif
  }

  /// Remote 64-bit atomic add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(uint64_t addr, uint32_t key, int64_t value) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (optimized()) {
      if (blueFlame()) {
        atomicAddBf(addr, key, value);
      } else {
        atomicAddDb(addr, key, value);
      }
    } else {
      if (blueFlame()) {
        atomicAddSharedBf(addr, key, value);
      } else {
        atomicAddBaseline(addr, key, value);
      }
    }
#else
    trap();
#endif
  }

  /// Wait until every work request posted so far on the queue pair has completed.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void flush([[maybe_unused]] int64_t maxSpinCount) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (optimized()) {
      flushExclusive(maxSpinCount);
    } else {
      flushShared(maxSpinCount);
    }
#else
    trap();
#endif
  }

 private:
#if defined(MSCCLPP_USE_GPUNETIO)
  // Each submission path is a separate non-inlined function so that selecting a path at run time does not merge the
  // register footprints of all paths into the calling kernel.

  // ---- MSCCLPP_GPUNETIO_BLUEFLAME=0, MSCCLPP_GPUNETIO_OPT=0: the initial GPUNetIO backend. ----

  MSCCLPP_GPUNETIO_PATH void putBaseline(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                         const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    __threadfence_system();
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        static_cast<doca_gpu_dev_verbs_qp*>(qp), {dst.addr + dstOffset, dst.key}, {src.addr + srcOffset, src.key}, size,
        &ticket);
  }

  MSCCLPP_GPUNETIO_PATH void putWithSignalBaseline(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                                   const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset,
                                                   uint64_t size) {
    __threadfence_system();
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        static_cast<doca_gpu_dev_verbs_qp*>(qp), {dst.addr + dstOffset, dst.key}, {src.addr + srcOffset, src.key}, size,
        {signalAddr, signalKey}, {atomicScratchAddr, atomicScratchKey}, 1, &ticket);
  }

  MSCCLPP_GPUNETIO_PATH void atomicAddBaseline(uint64_t addr, uint32_t key, int64_t value) {
    __threadfence_system();
    const doca_gpu_dev_verbs_addr remote{addr, key};
    const doca_gpu_dev_verbs_addr scratch{atomicScratchAddr, atomicScratchKey};
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        static_cast<doca_gpu_dev_verbs_qp*>(qp), remote, scratch, 0, remote, scratch, static_cast<uint64_t>(value),
        &ticket);
  }

  MSCCLPP_GPUNETIO_PATH void flushShared([[maybe_unused]] int64_t maxSpinCount) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t ticket =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_rsvd_index);
    if (ticket == 0) return;
    int status;
    [[maybe_unused]] int64_t spinCount = 0;
    while ((status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, ticket - 1)) ==
           EBUSY) {
      MSCCLPP_ASSERT_DEVICE((maxSpinCount < 0 || spinCount++ != maxSpinCount), "GPUNetIO flush timed out");
    }
    if (status != 0) {
      MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO completion queue reported an error");
      trap();
    }
  }

  // ---- MSCCLPP_GPUNETIO_BLUEFLAME=1, MSCCLPP_GPUNETIO_OPT=1: exclusive QP, BlueFlame with overlap. ----

  MSCCLPP_GPUNETIO_PATH void signalBf() {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
    const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      prepareSignalWqe(q, sharedWqe, index, token);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
  }

  MSCCLPP_GPUNETIO_PATH void putWithSignalBf(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                             const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    if (size > 0 && size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
      constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 2);
      withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
        if (size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) {
          doca_gpu_dev_verbs_wqe_prepare_write_inl(q, &sharedWqe[0], index, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                   dst.addr + dstOffset, dst.key, src.addr + srcOffset, size);
          submitInlineFromSharedNoWait(q, &sharedWqe[0]);
        } else {
          doca_gpu_dev_verbs_wqe_prepare_write(q, &sharedWqe[0], index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dst.addr + dstOffset,
                                               dst.key, src.addr + srcOffset, src.key, size);
          submitDmaFromSharedNoWait(q, &sharedWqe[0]);
        }

        const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
        copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), &sharedWqe[0]);
        prepareSignalWqe(q, &sharedWqe[1], index + 1, token);
        copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1), &sharedWqe[1]);
        doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index + 1);
        submitPreparedFromShared(q, index + 1, &sharedWqe[1]);
      });
      return;
    }
    putImpl(dst, dstOffset, src, srcOffset, size);
    signalBf();
  }

  MSCCLPP_GPUNETIO_PATH void atomicAddBf(uint64_t addr, uint32_t key, int64_t value) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      doca_gpu_dev_verbs_wqe_prepare_atomic(q, sharedWqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, addr, key, atomicScratchAddr,
                                            atomicScratchKey, sizeof(uint64_t), static_cast<uint64_t>(value), 0);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
  }

  MSCCLPP_GPUNETIO_PATH void flushExclusive([[maybe_unused]] int64_t maxSpinCount) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t ticket =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            &q->sq_rsvd_index);
    if (ticket == 0) return;
    int status;
    [[maybe_unused]] int64_t spinCount = 0;
    while ((status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                                       DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(
                q, ticket - 1)) == EBUSY) {
      MSCCLPP_ASSERT_DEVICE((maxSpinCount < 0 || spinCount++ != maxSpinCount), "GPUNetIO flush timed out");
    }
    if (status != 0) {
      MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO completion queue reported an error");
      trap();
    }
  }
  MSCCLPP_GPUNETIO_PATH void putImpl(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                     const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      doca_gpu_dev_verbs_ticket_t ticket;
      doca_gpu_dev_verbs_put<sharing, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                             DOCA_GPUNETIO_VERBS_CQ_64B>(q, {dst.addr + dstOffset, dst.key},
                                                         {src.addr + srcOffset, src.key}, size, &ticket,
                                                         DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
      submit(ticket);
      return;
    }

    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      if (size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) {
        // Inline copies the payload into the WQE; the NIC does not DMA-read the original source buffer.
        doca_gpu_dev_verbs_wqe_prepare_write_inl(q, sharedWqe, index, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                 dst.addr + dstOffset, dst.key, src.addr + srcOffset, size);
      } else {
        doca_gpu_dev_verbs_wqe_prepare_write(q, sharedWqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                             DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dst.addr + dstOffset, dst.key,
                                             src.addr + srcOffset, src.key, size);
      }
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
  }

  MSCCLPP_DEVICE_INLINE void prepareSignalWqe(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqePtr, uint16_t wqeIdx,
                                              uint64_t token) const {
    constexpr auto ctrlFlags = static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE | DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_STRONG_ORDERING);
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(q, wqePtr, wqeIdx, ctrlFlags, signalAddr, signalKey,
                                                         sizeof(uint64_t));
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<uint64_t>(q, wqePtr, token);
  }

  MSCCLPP_DEVICE_INLINE static void copyWqeToGlobal(doca_gpu_dev_verbs_wqe* dstGlobal,
                                                    const doca_gpu_dev_verbs_wqe* src) {
    auto* dst = reinterpret_cast<uint64_t*>(dstGlobal);
    auto* s = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(src));
    doca_gpu_dev_verbs_store_wqe_seg(dst + 0, s + 0);
    doca_gpu_dev_verbs_store_wqe_seg(dst + 2, s + 2);
    doca_gpu_dev_verbs_store_wqe_seg(dst + 4, s + 4);
    doca_gpu_dev_verbs_store_wqe_seg(dst + 6, s + 6);
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>
  MSCCLPP_DEVICE_INLINE static void submitPreparedFromShared(doca_gpu_dev_verbs_qp* q, uint64_t index,
                                                             doca_gpu_dev_verbs_wqe* sharedWqe) {
#if MSCCLPP_BULK_AVAILABLE
    bulkFence();
    doca_gpu_dev_verbs_submit_bf<Sharing, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, index + 1, sharedWqe);
    bulkStoreCommit();
    bulkStoreWaitSource();
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

  MSCCLPP_DEVICE_INLINE static void submitDmaFromSharedNoWait(doca_gpu_dev_verbs_qp* q,
                                                              doca_gpu_dev_verbs_wqe* sharedWqe) {
#if MSCCLPP_BULK_AVAILABLE
    bulkFence();
    doca_gpu_dev_verbs_ring_bf<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, sharedWqe);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

  MSCCLPP_DEVICE_INLINE static void submitInlineFromSharedNoWait(doca_gpu_dev_verbs_qp* q,
                                                                 doca_gpu_dev_verbs_wqe* sharedWqe) {
#if MSCCLPP_BULK_AVAILABLE
    bulkFence();
    void* bfPtr = reinterpret_cast<void*>(doca_gpu_dev_verbs_load_const(reinterpret_cast<uintptr_t*>(&q->sq_db)));
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 64;" : : "l"(bfPtr), "l"(sharedWqe));
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

#if MSCCLPP_BULK_AVAILABLE
  // One workspace per block, shared by all operations, so each operation does not add its own static allocation.
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_wqe* sharedWqeWorkspace() {
    __shared__ __align__(16) doca_gpu_dev_verbs_wqe sharedWqes[128];
    return sharedWqes;
  }

  // The calling thread's dedicated 2-WQE slot pair, or nullptr if it must lock a hashed slot pair.
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_wqe* dedicatedSharedWqes() {
    const unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned numThreads = blockDim.x * blockDim.y * blockDim.z;
    // Thread 0 and blocks with <= 64 threads have dedicated 2-WQE slots and need no workspace lock.
    return (tid == 0 || numThreads <= 64) ? &sharedWqeWorkspace()[tid * 2] : nullptr;
  }
#endif

  template <typename Fn>
  MSCCLPP_DEVICE_INLINE static void withSharedWqe(Fn&& fn) {
#if MSCCLPP_BULK_AVAILABLE
    if (auto* dedicated = dedicatedSharedWqes()) {
      fn(dedicated);
      return;
    }

    const unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned slot = 1 + ((tid - 1) % 63);
    uint64_t gridId;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(gridId));
    const uint64_t block = blockIdx.x + uint64_t(gridDim.x) * (blockIdx.y + uint64_t(gridDim.y) * blockIdx.z);
    auto* lock = &detail::gpuNetIoWqeLocks[(gridId ^ (block * 67) ^ slot) & 1023];
    bool submitted = false;
    do {
      if (atomicCAS(lock, 0u, 1u) == 0u) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        fn(&sharedWqeWorkspace()[slot * 2]);
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        atomicExch(lock, 0u);
        submitted = true;
      }
    } while (!submitted);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>
  MSCCLPP_DEVICE_INLINE void submit(uint64_t index) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      doca_gpu_dev_verbs_memcpy_data(sharedWqe, doca_gpu_dev_verbs_get_wqe_ptr(q, index), sizeof(*sharedWqe));
      submitPreparedFromShared<Sharing>(q, index, sharedWqe);
    });
  }

  // ---- MSCCLPP_GPUNETIO_BLUEFLAME=0, MSCCLPP_GPUNETIO_OPT=1: exclusive QP with regular doorbells. ----

  MSCCLPP_DEVICE_INLINE static void preparePayloadWqe(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqe,
                                                      uint64_t index, const GpuNetIoMemoryDeviceHandle& dst,
                                                      uint64_t dstOffset, const GpuNetIoMemoryDeviceHandle& src,
                                                      uint64_t srcOffset, uint64_t size) {
    if (size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) {
      doca_gpu_dev_verbs_wqe_prepare_write_inl(q, wqe, index, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                               dst.addr + dstOffset, dst.key, src.addr + srcOffset, size);
    } else {
      doca_gpu_dev_verbs_wqe_prepare_write(q, wqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dst.addr + dstOffset, dst.key,
                                           src.addr + srcOffset, src.key, size);
    }
  }

  // Publish WQEs [first, last], already written to the global SQ, and ring the regular doorbell.
  MSCCLPP_DEVICE_INLINE static void submitDb(doca_gpu_dev_verbs_qp* q, uint64_t first, uint64_t last) {
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, first, last);
    doca_gpu_dev_verbs_submit<sharing, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
        q, last + 1);
  }

  MSCCLPP_DEVICE_INLINE static uint64_t reserveExclusive(doca_gpu_dev_verbs_qp* q, uint32_t count) {
    return doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                               DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, count);
  }

  MSCCLPP_GPUNETIO_PATH void putDb(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                   const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      doca_gpu_dev_verbs_ticket_t ticket;
      doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                             DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                             DOCA_GPUNETIO_VERBS_CQ_64B>(q, {dst.addr + dstOffset, dst.key},
                                                         {src.addr + srcOffset, src.key}, size, &ticket);
      return;
    }
    const uint64_t index = reserveExclusive(q, 1);
    preparePayloadWqe(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index), index, dst, dstOffset, src, srcOffset, size);
    submitDb(q, index, index);
  }

  MSCCLPP_GPUNETIO_PATH void signalDb() {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t index = reserveExclusive(q, 1);
    const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
    prepareSignalWqe(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index), index, token);
    submitDb(q, index, index);
  }

  MSCCLPP_GPUNETIO_PATH void putWithSignalDb(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                             const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      putDb(dst, dstOffset, src, srcOffset, size);
      signalDb();
      return;
    }
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t index = reserveExclusive(q, 2);
    preparePayloadWqe(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index), index, dst, dstOffset, src, srcOffset, size);
    const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
    prepareSignalWqe(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1), index + 1, token);
    submitDb(q, index, index + 1);
  }

  MSCCLPP_GPUNETIO_PATH void atomicAddDb(uint64_t addr, uint32_t key, int64_t value) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t index = reserveExclusive(q, 1);
    doca_gpu_dev_verbs_wqe_prepare_atomic(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index), index,
                                          DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, addr, key, atomicScratchAddr,
                                          atomicScratchKey, sizeof(uint64_t), static_cast<uint64_t>(value), 0);
    submitDb(q, index, index);
  }

  // ---- MSCCLPP_GPUNETIO_BLUEFLAME=1, MSCCLPP_GPUNETIO_OPT=0: BlueFlame on DOCA's shared (GPU) QP mode. ----
  // Payload and signal are submitted together (no overlap): an unlocked payload BlueFlame write would race with
  // other threads' BlueFlame writes in shared mode.

  MSCCLPP_DEVICE_INLINE static uint64_t reserveShared(doca_gpu_dev_verbs_qp* q, uint32_t count) {
    return doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_QP_SQ,
                                               DOCA_GPUNETIO_VERBS_CQ_64B>(q, count);
  }

  // Prepare WQEs [first, last] (at most two) with `prepare(wqe, index)` and submit them through BlueFlame. With a
  // dedicated slot pair the WQEs are built in shared memory and copied to the SQ. Otherwise they are built in the SQ
  // and the last one is read back under the slot lock: shared-mode mark_wqes_ready waits for earlier reservations, so
  // it must not run while holding a slot lock that an earlier reserver may need.
  template <typename Prepare>
  MSCCLPP_DEVICE_INLINE void submitSharedBf(doca_gpu_dev_verbs_qp* q, uint64_t first, uint64_t last,
                                            Prepare&& prepare) {
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
#if MSCCLPP_BULK_AVAILABLE
    if (auto* sharedWqe = dedicatedSharedWqes()) {
      for (uint64_t i = first; i <= last; i++) {
        prepare(&sharedWqe[i - first], i);
        copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, i), &sharedWqe[i - first]);
      }
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, first, last);
      submitPreparedFromShared<sharing>(q, last, &sharedWqe[last - first]);
      return;
    }
#endif
    for (uint64_t i = first; i <= last; i++) prepare(doca_gpu_dev_verbs_get_wqe_ptr(q, i), i);
    doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, first, last);
    submit<sharing>(last);
  }

  MSCCLPP_GPUNETIO_PATH void putSharedBf(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                         const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      doca_gpu_dev_verbs_ticket_t ticket;
      doca_gpu_dev_verbs_put<sharing, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                             DOCA_GPUNETIO_VERBS_CQ_64B>(q, {dst.addr + dstOffset, dst.key},
                                                         {src.addr + srcOffset, src.key}, size, &ticket,
                                                         DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
      submit<sharing>(ticket);
      return;
    }
    const uint64_t index = reserveShared(q, 1);
    submitSharedBf(q, index, index, [&](doca_gpu_dev_verbs_wqe* wqe, uint64_t i) {
      doca_gpu_dev_verbs_wqe_prepare_write(q, wqe, i, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dst.addr + dstOffset, dst.key,
                                           src.addr + srcOffset, src.key, size);
    });
  }

  MSCCLPP_GPUNETIO_PATH void putWithSignalSharedBf(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                                   const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset,
                                                   uint64_t size) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      doca_gpu_dev_verbs_ticket_t ticket;
      doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, sharing,
                                    DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                                    DOCA_GPUNETIO_VERBS_CQ_64B>(
          q, {dst.addr + dstOffset, dst.key}, {src.addr + srcOffset, src.key}, size, {signalAddr, signalKey},
          {atomicScratchAddr, atomicScratchKey}, 1, &ticket, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
      submit<sharing>(ticket);
      return;
    }
    const uint64_t index = reserveShared(q, 2);
    submitSharedBf(q, index, index + 1, [&](doca_gpu_dev_verbs_wqe* wqe, uint64_t i) {
      if (i == index) {
        doca_gpu_dev_verbs_wqe_prepare_write(q, wqe, i, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                             DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dst.addr + dstOffset, dst.key,
                                             src.addr + srcOffset, src.key, size);
      } else {
        doca_gpu_dev_verbs_wqe_prepare_atomic(q, wqe, i, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                              DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, signalAddr, signalKey,
                                              atomicScratchAddr, atomicScratchKey, sizeof(uint64_t), 1, 0);
      }
    });
  }

  // Same work requests as the baseline zero-byte put_signal: a NOP followed by the atomic.
  MSCCLPP_GPUNETIO_PATH void atomicAddSharedBf(uint64_t addr, uint32_t key, int64_t value) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t index = reserveShared(q, 2);
    submitSharedBf(q, index, index + 1, [&](doca_gpu_dev_verbs_wqe* wqe, uint64_t i) {
      if (i == index) {
        doca_gpu_dev_verbs_wqe_prepare_nop(q, wqe, i, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
      } else {
        doca_gpu_dev_verbs_wqe_prepare_atomic(q, wqe, i, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                              DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, addr, key, atomicScratchAddr,
                                              atomicScratchKey, sizeof(uint64_t), static_cast<uint64_t>(value), 0);
      }
    });
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#if defined(MSCCLPP_GPUNETIO_PATH)
#undef MSCCLPP_GPUNETIO_PATH
#endif

#endif  // MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
