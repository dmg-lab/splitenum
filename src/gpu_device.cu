#include "isotopy_gpu.cu"

#include "gpurefine.h"

namespace SplitEnum {
static_assert(sizeof(IsotopyGPU::Tables) + sizeof(SplitEnum::RefineTables) +
                      IsotopyGPU::kWarpsPerBlock *
                          (sizeof(IsotopyGPU::Scratch) + sizeof(RefineScratch) +
                           2 * IsotopyGPU::kMaxQ1Edges) <=
                  48 * 1024,
              "refine_classify_kernel shared memory exceeds 48 KB");

__global__ void refine_classify_kernel(const uint16_t* ids,
                                       const uint32_t* offsets,
                                       const uint64_t* signs,
                                       IsotopyGPU::Result* out,
                                       uint8_t* edges_out, int batch,
                                       int edge_stride, int sign_stride,
                                       RefineTables rt) {
  using namespace IsotopyGPU;
  __shared__ Tables tables;
  __shared__ RefineTables rtab;
  __shared__ Scratch scratch[kWarpsPerBlock];
  __shared__ RefineScratch rscratch[kWarpsPerBlock];
  __shared__ uint8_t edges[kWarpsPerBlock][2 * kMaxQ1Edges];

  uint32_t* dst = reinterpret_cast<uint32_t*>(&tables);
  const uint32_t* src = reinterpret_cast<const uint32_t*>(&g_tables);
  for (int i = threadIdx.x; i < (int)(sizeof(Tables) / sizeof(uint32_t));
       i += blockDim.x)
    dst[i] = src[i];
  if (threadIdx.x == 0) rtab = rt;
  __syncthreads();

  const int slot = threadIdx.x >> 5;
  const int instance = (int)blockIdx.x * kWarpsPerBlock + slot;
  if (instance >= batch) return;

  const uint32_t at = offsets[instance];
  const int n = (int)(offsets[instance + 1] - at);
  Result r;
  refine_classify_one(tables, rtab, ids + at, n,
                      signs + (size_t)instance * sign_stride, rscratch[slot],
                      edges[slot], scratch[slot], Warp<32>{}, r);

  const int lane = threadIdx.x & 31;
  if (edges_out != nullptr && r.flags != kRefineFailed)
    for (int i = lane; i < edge_stride; i += 32)
      edges_out[(size_t)instance * edge_stride + i] = edges[slot][i];
  if (lane == 0) out[instance] = r;
}
}

#ifndef __CUDA_ARCH__

#include "gpudevice.h"

namespace SplitEnum {
struct GpuDevice {
  int device = 0;
  cudaStream_t stream = nullptr;
  IsotopyGPU::DeviceBatch batch;
  bool refine = false;
  RefineTables rt{};
  uint64_t* d_forced = nullptr;
  uint64_t* d_blocked = nullptr;
  uint64_t* d_cross = nullptr;
  uint8_t* d_pool = nullptr;
  uint16_t* d_ids = nullptr;
  uint32_t* d_offsets = nullptr;
};

namespace {
template <typename T>
T* upload(const T* host, size_t n) {
  T* d = nullptr;
  ISO_CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
  ISO_CUDA_CHECK(cudaMemcpy(d, host, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
}

int gpu_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

GpuDevice* gpu_device_open(int device, int delta, int capacity,
                           const RefineTables* refine, size_t ids_capacity) {
  ISO_CUDA_CHECK(cudaSetDevice(device));
  GpuDevice* d = new GpuDevice;
  d->device = device;
  ISO_CUDA_CHECK(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking));
  const IsotopyGPU::Tables t = IsotopyGPU::build_tables(delta);
  IsotopyGPU::device_batch_alloc(d->batch, t, capacity);
  if (refine != nullptr) {
    d->refine = true;
    d->rt = *refine;
    const size_t rows = (size_t)refine->nsplits * kRefineWords;
    d->d_forced = upload(refine->forced, rows);
    d->d_blocked = upload(refine->blocked, rows);
    d->d_cross = upload(refine->cross, (size_t)refine->pool_size * kRefineWords);
    d->d_pool = upload(refine->pool, 2 * (size_t)refine->pool_size);
    d->rt.forced = d->d_forced;
    d->rt.blocked = d->d_blocked;
    d->rt.cross = d->d_cross;
    d->rt.pool = d->d_pool;
    ISO_CUDA_CHECK(cudaMalloc(&d->d_ids, ids_capacity * sizeof(uint16_t)));
    ISO_CUDA_CHECK(
        cudaMalloc(&d->d_offsets, ((size_t)capacity + 1) * sizeof(uint32_t)));
  }
  return d;
}

void gpu_device_launch(GpuDevice* d, const uint8_t* edges,
                       const uint64_t* signs, int batch, void* results) {
  using namespace IsotopyGPU;
  ISO_CUDA_CHECK(cudaSetDevice(d->device));
  DeviceBatch& b = d->batch;
  ISO_CUDA_CHECK(cudaMemcpyAsync(b.edges, edges,
                                 (size_t)batch * b.edge_stride * sizeof(uint8_t),
                                 cudaMemcpyHostToDevice, d->stream));
  ISO_CUDA_CHECK(cudaMemcpyAsync(b.signs, signs,
                                 (size_t)batch * b.sign_stride * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, d->stream));
  const int blocks = (batch + kWarpsPerBlock - 1) / kWarpsPerBlock;
  classify_kernel<<<blocks, kThreadsPerBlock, 0, d->stream>>>(
      b.edges, b.signs, b.out, batch, b.edge_stride, b.sign_stride, b.keys,
      b.nkeys);
  ISO_CUDA_CHECK(cudaGetLastError());
  ISO_CUDA_CHECK(cudaMemcpyAsync(results, b.out, (size_t)batch * sizeof(Result),
                                 cudaMemcpyDeviceToHost, d->stream));
}

void gpu_device_launch_ids(GpuDevice* d, const uint64_t* signs,
                           const uint32_t* offsets, const uint16_t* ids,
                           int batch, size_t nids, void* results,
                           uint8_t* edges_out) {
  using namespace IsotopyGPU;
  ISO_CUDA_CHECK(cudaSetDevice(d->device));
  DeviceBatch& b = d->batch;
  ISO_CUDA_CHECK(cudaMemcpyAsync(b.signs, signs,
                                 (size_t)batch * b.sign_stride * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, d->stream));
  ISO_CUDA_CHECK(cudaMemcpyAsync(d->d_offsets, offsets,
                                 ((size_t)batch + 1) * sizeof(uint32_t),
                                 cudaMemcpyHostToDevice, d->stream));
  ISO_CUDA_CHECK(cudaMemcpyAsync(d->d_ids, ids, nids * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, d->stream));
  const int blocks = (batch + kWarpsPerBlock - 1) / kWarpsPerBlock;
  refine_classify_kernel<<<blocks, kThreadsPerBlock, 0, d->stream>>>(
      d->d_ids, d->d_offsets, b.signs, b.out,
      edges_out != nullptr ? b.edges : nullptr, batch, b.edge_stride,
      b.sign_stride, d->rt);
  ISO_CUDA_CHECK(cudaGetLastError());
  ISO_CUDA_CHECK(cudaMemcpyAsync(results, b.out, (size_t)batch * sizeof(Result),
                                 cudaMemcpyDeviceToHost, d->stream));
  if (edges_out != nullptr)
    ISO_CUDA_CHECK(cudaMemcpyAsync(edges_out, b.edges,
                                   (size_t)batch * b.edge_stride,
                                   cudaMemcpyDeviceToHost, d->stream));
}

void gpu_device_wait(GpuDevice* d) {
  ISO_CUDA_CHECK(cudaStreamSynchronize(d->stream));
}

void gpu_device_close(GpuDevice* d) {
  if (d == nullptr) return;
  ISO_CUDA_CHECK(cudaSetDevice(d->device));
  ISO_CUDA_CHECK(cudaStreamSynchronize(d->stream));
  IsotopyGPU::device_batch_free(d->batch);
  if (d->refine) {
    ISO_CUDA_CHECK(cudaFree(d->d_forced));
    ISO_CUDA_CHECK(cudaFree(d->d_blocked));
    ISO_CUDA_CHECK(cudaFree(d->d_cross));
    ISO_CUDA_CHECK(cudaFree(d->d_pool));
    ISO_CUDA_CHECK(cudaFree(d->d_ids));
    ISO_CUDA_CHECK(cudaFree(d->d_offsets));
  }
  ISO_CUDA_CHECK(cudaStreamDestroy(d->stream));
  delete d;
}

void* gpu_host_alloc(size_t bytes) {
  void* p = nullptr;
  ISO_CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
  return p;
}

void gpu_host_free(void* p) {
  if (p != nullptr) ISO_CUDA_CHECK(cudaFreeHost(p));
}
}

#endif
