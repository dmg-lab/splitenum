#pragma once

#include <cstddef>
#include <cstdint>

namespace SplitEnum {
struct GpuDevice;
struct RefineTables;

int gpu_device_count();
GpuDevice* gpu_device_open(int device, int delta, int capacity,
                           const RefineTables* refine, size_t ids_capacity);
void gpu_device_launch(GpuDevice* d, const uint8_t* edges,
                       const uint64_t* signs, int batch, void* results);
void gpu_device_launch_ids(GpuDevice* d, const uint64_t* signs,
                           const uint32_t* offsets, const uint16_t* ids,
                           int batch, size_t nids, void* results,
                           uint8_t* edges_out);
void gpu_device_wait(GpuDevice* d);
void gpu_device_close(GpuDevice* d);
void* gpu_host_alloc(size_t bytes);
void gpu_host_free(void* p);
}
