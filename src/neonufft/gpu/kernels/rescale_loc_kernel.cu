#include "neonufft//config.h"
//---
#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>

#include "neonufft/enums.h"
#include "neonufft/gpu/kernels/rescale_loc_kernel.hpp"
#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/gpu/util/kernel_launch_grid.hpp"
#include "neonufft/gpu/util/partition_group.hpp"
#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/util/math.hpp"
#include "neonufft/util/point.hpp"

namespace neonufft {
namespace gpu {

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    compute_part_sizes_1d_kernel(ConstDeviceView<T, 1> loc, IntType offset, IntType grid_size,
                                 DeviceView<PartitionGroup, 1> partition) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc.size();
       idx += gridDim.x * BLOCK_SIZE) {
    auto l = (loc[idx] + math::pi<T>)*two_pi_inv;
    if constexpr (std::is_same_v<T, float>) {
      l = l - floorf(l);
    } else {
      l = l - floor(l);
    }

    IntType idx_part = (IntType(l * grid_size) + offset) / PartitionGroup::width;
    idx_part =  min(idx_part, partition.shape(0) - 1);
    atomicAdd(&(partition[idx_part].size), 1);
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    compute_part_sizes_2d_kernel(ConstDeviceView<T, 1> loc_x, ConstDeviceView<T, 1> loc_y,
                                 IntType offset_x, IntType offset_y, IntType grid_size_x,
                                 IntType grid_size_y, DeviceView<PartitionGroup, 2> partition) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc_x.size();
       idx += gridDim.x * BLOCK_SIZE) {
    auto l_x = (loc_x[idx] + math::pi<T>)*two_pi_inv;
    auto l_y = (loc_y[idx] + math::pi<T>)*two_pi_inv;
    if constexpr (std::is_same_v<T, float>) {
      l_x = l_x - floorf(l_x);
      l_y = l_y - floorf(l_y);
    } else {
      l_x = l_x - floor(l_x);
      l_y = l_y - floor(l_y);
    }

    IntType idx_part_x = (IntType(l_x * grid_size_x) + offset_x) / PartitionGroup::width;
    IntType idx_part_y = (IntType(l_y * grid_size_y) + offset_y) / PartitionGroup::width;
    idx_part_x =  min(idx_part_x, partition.shape(0) - 1);
    idx_part_y =  min(idx_part_y, partition.shape(1) - 1);

    atomicAdd(&(partition[{idx_part_x, idx_part_y}].size), 1);
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    compute_part_sizes_3d_kernel(ConstDeviceView<T, 1> loc_x, ConstDeviceView<T, 1> loc_y,
                                 ConstDeviceView<T, 1> loc_z, IntType offset_x, IntType offset_y,
                                 IntType offset_z, IntType grid_size_x, IntType grid_size_y,
                                 IntType grid_size_z, DeviceView<PartitionGroup, 3> partition) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc_x.size();
       idx += gridDim.x * BLOCK_SIZE) {
    auto l_x = (loc_x[idx] + math::pi<T>)*two_pi_inv;
    auto l_y = (loc_y[idx] + math::pi<T>)*two_pi_inv;
    auto l_z = (loc_z[idx] + math::pi<T>)*two_pi_inv;
    if constexpr (std::is_same_v<T, float>) {
      l_x = l_x - floorf(l_x);
      l_y = l_y - floorf(l_y);
      l_z = l_z - floorf(l_z);
    } else {
      l_x = l_x - floor(l_x);
      l_y = l_y - floor(l_y);
      l_z = l_z - floor(l_z);
    }

    IntType idx_part_x = (IntType(l_x * grid_size_x) + offset_x) / PartitionGroup::width;
    IntType idx_part_y = (IntType(l_y * grid_size_y) + offset_y) / PartitionGroup::width;
    IntType idx_part_z = (IntType(l_z * grid_size_z) + offset_z) / PartitionGroup::width;

    idx_part_x =  min(idx_part_x, partition.shape(0) - 1);
    idx_part_y =  min(idx_part_y, partition.shape(1) - 1);
    idx_part_z =  min(idx_part_y, partition.shape(2) - 1);

    atomicAdd(&(partition[{idx_part_x, idx_part_y, idx_part_z}].size), 1);
  }
}

template <typename T>
__global__ static void compute_part_offsets_kernel(DeviceView<PartitionGroup, 1> partition) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  __shared__ unsigned long long  offset_counter;

  if (threadIdx.x == 0) {
    offset_counter = 0;
  }
  __syncthreads();

  for (IntType idx = threadIdx.x; idx < partition.size(); idx += blockDim.x) {
    const auto size = partition[idx].size;
    if (size) {
      partition[idx].begin = atomicAdd(&offset_counter, size);
      partition[idx].size = 0;
    }
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    rescale_and_permut_1d_kernel(ConstDeviceView<T, 1> loc, IntType offset, IntType grid_size,
                                 DeviceView<PartitionGroup, 1> partition,
                                 DeviceView<Point<T, 1>, 1> points) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc.size();
       idx += gridDim.x * BLOCK_SIZE) {
    auto l = (loc[idx] + math::pi<T>)*two_pi_inv;
    if constexpr (std::is_same_v<T, float>) {
      l = l - floorf(l);
    } else {
      l = l - floor(l);
    }

    IntType idx_part = (IntType(l * grid_size) + offset) / PartitionGroup::width;
    idx_part = min(idx_part, partition.shape(0) - 1);
    const auto local_offset = atomicAdd(&(partition[idx_part].size), 1);

    Point<T, 1> p;
    p.coord[0] = l;
    p.index = idx;

    points[partition[idx_part].begin + local_offset] = p;
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    rescale_and_permut_2d_kernel(ConstDeviceView<T, 1> loc_x, ConstDeviceView<T, 1> loc_y,
                                 IntType offset_x, IntType offset_y, IntType grid_size_x,
                                 IntType grid_size_y, DeviceView<PartitionGroup, 2> partition,
                                 DeviceView<Point<T, 2>, 1> points) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc_x.size();
       idx += gridDim.x * BLOCK_SIZE) {
    auto l_x = (loc_x[idx] + math::pi<T>)*two_pi_inv;
    auto l_y = (loc_y[idx] + math::pi<T>)*two_pi_inv;
    if constexpr (std::is_same_v<T, float>) {
      l_x = l_x - floorf(l_x);
      l_y = l_y - floorf(l_y);
    } else {
      l_x = l_x - floor(l_x);
      l_y = l_y - floor(l_y);
    }

    IntType idx_part_x = (IntType(l_x * grid_size_x) + offset_x) / PartitionGroup::width;
    IntType idx_part_y = (IntType(l_y * grid_size_y) + offset_y) / PartitionGroup::width;
    idx_part_x =  min(idx_part_x, partition.shape(0) - 1);
    idx_part_y =  min(idx_part_y, partition.shape(1) - 1);

    const auto local_offset = atomicAdd(&(partition[{idx_part_x, idx_part_y}].size), 1);

    Point<T, 2> p;
    p.coord[0] = l_x;
    p.coord[1] = l_y;
    p.index = idx;

    points[partition[{idx_part_x, idx_part_y}].begin + local_offset] = p;
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    rescale_and_permut_3d_kernel(ConstDeviceView<T, 1> loc_x, ConstDeviceView<T, 1> loc_y,
                                 ConstDeviceView<T, 1> loc_z, IntType offset_x, IntType offset_y,
                                 IntType offset_z, IntType grid_size_x, IntType grid_size_y,
                                 IntType grid_size_z, DeviceView<PartitionGroup, 3> partition,
                                 DeviceView<Point<T, 3>, 1> points) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc_x.size();
       idx += gridDim.x * BLOCK_SIZE) {
    auto l_x = (loc_x[idx] + math::pi<T>)*two_pi_inv;
    auto l_y = (loc_y[idx] + math::pi<T>)*two_pi_inv;
    auto l_z = (loc_z[idx] + math::pi<T>)*two_pi_inv;
    if constexpr (std::is_same_v<T, float>) {
      l_x = l_x - floorf(l_x);
      l_y = l_y - floorf(l_y);
      l_z = l_z - floorf(l_z);
    } else {
      l_x = l_x - floor(l_x);
      l_y = l_y - floor(l_y);
      l_z = l_z - floor(l_z);
    }

    IntType idx_part_x = (IntType(l_x * grid_size_x) + offset_x) / PartitionGroup::width;
    IntType idx_part_y = (IntType(l_y * grid_size_y) + offset_y) / PartitionGroup::width;
    IntType idx_part_z = (IntType(l_z * grid_size_z) + offset_z) / PartitionGroup::width;

    idx_part_x =  min(idx_part_x, partition.shape(0) - 1);
    idx_part_y =  min(idx_part_y, partition.shape(1) - 1);
    idx_part_z =  min(idx_part_y, partition.shape(2) - 1);

    const auto local_offset = atomicAdd(&(partition[{idx_part_x, idx_part_y, idx_part_z}].size), 1);

    Point<T, 3> p;
    p.coord[0] = l_x;
    p.coord[1] = l_y;
    p.coord[2] = l_z;
    p.index = idx;

    points[partition[{idx_part_x, idx_part_y, idx_part_z}].begin + local_offset] = p;
  }
}

template <typename T, IntType DIM>
auto rescale_and_permut(const api::DevicePropType& prop, const api::StreamType& stream,
                        std::array<ConstDeviceView<T, 1>, DIM> loc, std::array<IntType, DIM> offset,
                        std::array<IntType, DIM> grid_size,
                        DeviceView<PartitionGroup, DIM> partition,
                        DeviceView<Point<T, DIM>, 1> points) -> void {
  assert(loc[0].size() == points.size());
  assert(partition.is_contiguous());
  constexpr int block_size = 512;
  const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {points.size(), 1, 1}, block);

  partition.zero(stream);

  if constexpr (DIM == 1) {
    api::launch_kernel(compute_part_sizes_1d_kernel<T, block_size>, grid, block, 0, stream, loc[0],
                       offset[0], grid_size[0], partition);

    api::launch_kernel(compute_part_offsets_kernel<T>, dim3{1, 1, 1}, dim3{1, 1, 1}, 0, stream,
                       partition);

    api::launch_kernel(rescale_and_permut_1d_kernel<T, block_size>, grid, block, 0, stream, loc[0],
                       offset[0], grid_size[0], partition, points);

  } else if constexpr (DIM == 2){
    api::launch_kernel(compute_part_sizes_2d_kernel<T, block_size>, grid, block, 0, stream, loc[0],
                       loc[1], offset[0], offset[1], grid_size[0], grid_size[1], partition);

    api::launch_kernel(compute_part_offsets_kernel<T>, dim3{1, 1, 1}, dim3{1, 1, 1}, 0, stream,
                       DeviceView<PartitionGroup, 1>(partition.data(), partition.size(), 1));

    api::launch_kernel(rescale_and_permut_2d_kernel<T, block_size>, grid, block, 0, stream, loc[0],
                       loc[1], offset[0], offset[1], grid_size[0], grid_size[1], partition, points);

  } else if constexpr (DIM == 3) {
    api::launch_kernel(compute_part_sizes_3d_kernel<T, block_size>, grid, block, 0, stream, loc[0],
                       loc[1], loc[2], offset[0], offset[1], offset[2], grid_size[0], grid_size[1],
                       grid_size[2], partition);

    api::launch_kernel(compute_part_offsets_kernel<T>, dim3{1, 1, 1}, dim3{1, 1, 1}, 0, stream,
                       DeviceView<PartitionGroup, 1>(partition.data(), partition.size(), 1));

    api::launch_kernel(rescale_and_permut_3d_kernel<T, block_size>, grid, block, 0, stream, loc[0],
                       loc[1], loc[2], offset[0], offset[1], offset[2], grid_size[0], grid_size[1],
                       grid_size[2], partition, points);
  } else {
    throw InternalError("invalid dimension");
  }
}

template auto rescale_and_permut<float, 1>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           std::array<ConstDeviceView<float, 1>, 1> loc,
                                           std::array<IntType, 1> offset,
                                           std::array<IntType, 1> grid_size,
                                           DeviceView<PartitionGroup, 1> partition,
                                           DeviceView<Point<float, 1>, 1> points) -> void;

template auto rescale_and_permut<float, 2>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           std::array<ConstDeviceView<float, 1>, 2> loc,
                                           std::array<IntType, 2> offset,
                                           std::array<IntType, 2> grid_size,
                                           DeviceView<PartitionGroup, 2> partition,
                                           DeviceView<Point<float, 2>, 1> points) -> void;

template auto rescale_and_permut<float, 3>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           std::array<ConstDeviceView<float, 1>, 3> loc,
                                           std::array<IntType, 3> offset,
                                           std::array<IntType, 3> grid_size,
                                           DeviceView<PartitionGroup, 3> partition,
                                           DeviceView<Point<float, 3>, 1> points) -> void;

template auto rescale_and_permut<double, 1>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           std::array<ConstDeviceView<double, 1>, 1> loc,
                                           std::array<IntType, 1> offset,
                                           std::array<IntType, 1> grid_size,
                                           DeviceView<PartitionGroup, 1> partition,
                                           DeviceView<Point<double, 1>, 1> points) -> void;

template auto rescale_and_permut<double, 2>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           std::array<ConstDeviceView<double, 1>, 2> loc,
                                           std::array<IntType, 2> offset,
                                           std::array<IntType, 2> grid_size,
                                           DeviceView<PartitionGroup, 2> partition,
                                           DeviceView<Point<double, 2>, 1> points) -> void;

template auto rescale_and_permut<double, 3>(const api::DevicePropType& prop,
                                            const api::StreamType& stream,
                                            std::array<ConstDeviceView<double, 1>, 3> loc,
                                            std::array<IntType, 3> offset,
                                            std::array<IntType, 3> grid_size,
                                            DeviceView<PartitionGroup, 3> partition,
                                            DeviceView<Point<double, 3>, 1> points) -> void;

}  // namespace gpu
}  // namespace neonufft
