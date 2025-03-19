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
#include "neonufft/util/stack_array.hpp"

namespace neonufft {
namespace gpu {

template <typename T, IntType DIM, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    compute_part_sizes_kernel(StackArray<ConstDeviceView<T, 1>, DIM> loc, IndexArray<DIM> grid_size,
                              DeviceView<PartitionGroup, DIM> partition) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc[0].size();
       idx += gridDim.x * BLOCK_SIZE) {
    T l[DIM];
    IndexArray<DIM> idx_part;
    for (IntType d = 0; d < DIM; ++d) {
      l[d] = (loc[d][idx] + math::pi<T>)*two_pi_inv;
      if constexpr (std::is_same_v<T, float>) {
        l[d] = l[d] - floorf(l[d]);
      } else {
        l[d] = l[d] - floor(l[d]);
      }
      idx_part[d] = IntType(l[d] * grid_size[d]) / PartitionGroup::width;
      idx_part[d] = min(idx_part[d], partition.shape(d) - 1);
    }

    atomicAdd(&(partition[idx_part].size), 1);
  }
}
template <typename T, IntType DIM, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    compute_part_sizes_t3_kernel(StackArray<ConstDeviceView<T, 1>, DIM> loc,
                                 IndexArray<DIM> grid_size, StackArray<T, DIM> offset,
                                 StackArray<T, DIM> scaling_factor,
                                 DeviceView<PartitionGroup, DIM> partition) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc[0].size();
       idx += gridDim.x * BLOCK_SIZE) {
    T l[DIM];
    IndexArray<DIM> idx_part;
    for (IntType d = 0; d < DIM; ++d) {
      l[d] = (loc[d][idx] - offset[d]) * scaling_factor[d];
      l[d] = (l[d] + math::pi<T>)*two_pi_inv;
      if constexpr (std::is_same_v<T, float>) {
        l[d] = l[d] - floorf(l[d]);
      } else {
        l[d] = l[d] - floor(l[d]);
      }
      idx_part[d] = IntType(l[d] * grid_size[d]) / PartitionGroup::width;
      idx_part[d] = min(idx_part[d], partition.shape(d) - 1);
    }

    atomicAdd(&(partition[idx_part].size), 1);
  }
}

template <typename T>
__global__ static void compute_part_offsets_kernel(DeviceView<PartitionGroup, 1> partition) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  __shared__ unsigned long long offset_counter;

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

template <typename T, IntType DIM, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    rescale_and_permut_kernel(StackArray<ConstDeviceView<T, 1>, DIM> loc, IndexArray<DIM> grid_size,
                              DeviceView<PartitionGroup, DIM> partition,
                              DeviceView<Point<T, DIM>, 1> points) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc[0].size();
       idx += gridDim.x * BLOCK_SIZE) {
    Point<T, DIM> p;
    p.index = idx;
    IndexArray<DIM> idx_part;
    for (IntType d = 0; d < DIM; ++d) {
      p.coord[d] = (loc[d][idx] + math::pi<T>)*two_pi_inv;
      if constexpr (std::is_same_v<T, float>) {
        p.coord[d] = p.coord[d] - floorf(p.coord[d]);
      } else {
        p.coord[d] = p.coord[d] - floor(p.coord[d]);
      }
      idx_part[d] = IntType(p.coord[d] * grid_size[d]) / PartitionGroup::width;
      idx_part[d] = min(idx_part[d], partition.shape(d) - 1);
    }

    const auto local_offset = atomicAdd(&(partition[idx_part].size), 1);
    points[partition[idx_part].begin + local_offset] = p;
  }
}

template <typename T, IntType DIM, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    rescale_and_permut_t3_kernel(StackArray<ConstDeviceView<T, 1>, DIM> loc, IndexArray<DIM> grid_size,
                              StackArray<T, DIM> offset, StackArray<T, DIM> scaling_factor,
                              DeviceView<PartitionGroup, DIM> partition,
                              DeviceView<Point<T, DIM>, 1> points) {
  constexpr T two_pi = 2 * math::pi<T>;
  constexpr T two_pi_inv = 1 / (2 * math::pi<T>);

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < loc[0].size();
       idx += gridDim.x * BLOCK_SIZE) {
    Point<T, DIM> p;
    p.index = idx;
    IndexArray<DIM> idx_part;
    for (IntType d = 0; d < DIM; ++d) {
      p.coord[d] = (loc[d][idx] - offset[d]) * scaling_factor[d];
      p.coord[d] = (p.coord[d] + math::pi<T>)*two_pi_inv;
      if constexpr (std::is_same_v<T, float>) {
        p.coord[d] = p.coord[d] - floorf(p.coord[d]);
      } else {
        p.coord[d] = p.coord[d] - floor(p.coord[d]);
      }
      idx_part[d] = IntType(p.coord[d] * grid_size[d]) / PartitionGroup::width;
      idx_part[d] = min(idx_part[d], partition.shape(d) - 1);
    }

    const auto local_offset = atomicAdd(&(partition[idx_part].size), 1);
    points[partition[idx_part].begin + local_offset] = p;
  }
}

template <typename T, IntType DIM>
auto rescale_and_permut(const api::DevicePropType& prop, const api::StreamType& stream,
                        StackArray<ConstDeviceView<T, 1>, DIM> loc,
                        IndexArray<DIM> grid_size,
                        DeviceView<PartitionGroup, DIM> partition,
                        DeviceView<Point<T, DIM>, 1> points) -> void {
  assert(loc[0].size() == points.size());
  assert(partition.is_contiguous());
  constexpr int block_size = 512;
  const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {points.size(), 1, 1}, block);

  partition.zero(stream);

  api::launch_kernel(compute_part_sizes_kernel<T, DIM, block_size>, grid, block, 0, stream, loc,
                     grid_size, partition);

  assert(partition.is_contiguous());
  api::launch_kernel(compute_part_offsets_kernel<T>, dim3{1, 1, 1},
                     dim3(prop.maxThreadsDim[0], 1, 1), 0, stream,
                     DeviceView<PartitionGroup, 1>(partition.data(), partition.size(), 1));

  api::launch_kernel(rescale_and_permut_kernel<T, DIM, block_size>, grid, block, 0, stream, loc,
                     grid_size, partition, points);
}

template <typename T, IntType DIM>
auto rescale_and_permut_t3(const api::DevicePropType& prop, const api::StreamType& stream,
                           StackArray<ConstDeviceView<T, 1>, DIM> loc, IndexArray<DIM> grid_size,
                           StackArray<T, DIM> offset, StackArray<T, DIM> scaling_factor,
                           DeviceView<PartitionGroup, DIM> partition,
                           DeviceView<Point<T, DIM>, 1> points) -> void {
  assert(loc[0].size() == points.size());
  assert(partition.is_contiguous());
  constexpr int block_size = 512;
  const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {points.size(), 1, 1}, block);

  partition.zero(stream);

  api::launch_kernel(compute_part_sizes_t3_kernel<T, DIM, block_size>, grid, block, 0, stream, loc,
                     grid_size, offset, scaling_factor, partition);

  assert(partition.is_contiguous());
  api::launch_kernel(compute_part_offsets_kernel<T>, dim3{1, 1, 1},
                     dim3(prop.maxThreadsDim[0], 1, 1), 0, stream,
                     DeviceView<PartitionGroup, 1>(partition.data(), partition.size(), 1));

  api::launch_kernel(rescale_and_permut_t3_kernel<T, DIM, block_size>, grid, block, 0, stream, loc,
                     grid_size, offset,scaling_factor, partition, points);
}

template auto rescale_and_permut<float, 1>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           StackArray<ConstDeviceView<float, 1>, 1> loc,
                                           IndexArray<1> grid_size,
                                           DeviceView<PartitionGroup, 1> partition,
                                           DeviceView<Point<float, 1>, 1> points) -> void;

template auto rescale_and_permut<float, 2>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           StackArray<ConstDeviceView<float, 1>, 2> loc,
                                           IndexArray<2> grid_size,
                                           DeviceView<PartitionGroup, 2> partition,
                                           DeviceView<Point<float, 2>, 1> points) -> void;

template auto rescale_and_permut<float, 3>(const api::DevicePropType& prop,
                                           const api::StreamType& stream,
                                           StackArray<ConstDeviceView<float, 1>, 3> loc,
                                           IndexArray<3> grid_size,
                                           DeviceView<PartitionGroup, 3> partition,
                                           DeviceView<Point<float, 3>, 1> points) -> void;

template auto rescale_and_permut<double, 1>(const api::DevicePropType& prop,
                                            const api::StreamType& stream,
                                            StackArray<ConstDeviceView<double, 1>, 1> loc,
                                            IndexArray<1> grid_size,
                                            DeviceView<PartitionGroup, 1> partition,
                                            DeviceView<Point<double, 1>, 1> points) -> void;

template auto rescale_and_permut<double, 2>(const api::DevicePropType& prop,
                                            const api::StreamType& stream,
                                            StackArray<ConstDeviceView<double, 1>, 2> loc,
                                            IndexArray<2> grid_size,
                                            DeviceView<PartitionGroup, 2> partition,
                                            DeviceView<Point<double, 2>, 1> points) -> void;

template auto rescale_and_permut<double, 3>(const api::DevicePropType& prop,
                                            const api::StreamType& stream,
                                            StackArray<ConstDeviceView<double, 1>, 3> loc,
                                            IndexArray<3> grid_size,
                                            DeviceView<PartitionGroup, 3> partition,
                                            DeviceView<Point<double, 3>, 1> points) -> void;

template auto rescale_and_permut_t3<float, 1>(const api::DevicePropType& prop,
                                              const api::StreamType& stream,
                                              StackArray<ConstDeviceView<float, 1>, 1> loc,
                                              IndexArray<1> grid_size, StackArray<float, 1> offset,
                                              StackArray<float, 1> scaling_factor,
                                              DeviceView<PartitionGroup, 1> partition,
                                              DeviceView<Point<float, 1>, 1> points) -> void;

template auto rescale_and_permut_t3<float, 2>(const api::DevicePropType& prop,
                                              const api::StreamType& stream,
                                              StackArray<ConstDeviceView<float, 1>, 2> loc,
                                              IndexArray<2> grid_size, StackArray<float, 2> offset,
                                              StackArray<float, 2> scaling_factor,
                                              DeviceView<PartitionGroup, 2> partition,
                                              DeviceView<Point<float, 2>, 1> points) -> void;

template auto rescale_and_permut_t3<float, 3>(const api::DevicePropType& prop,
                                              const api::StreamType& stream,
                                              StackArray<ConstDeviceView<float, 1>, 3> loc,
                                              IndexArray<3> grid_size, StackArray<float, 3> offset,
                                              StackArray<float, 3> scaling_factor,
                                              DeviceView<PartitionGroup, 3> partition,
                                              DeviceView<Point<float, 3>, 1> points) -> void;

template auto rescale_and_permut_t3<double, 1>(const api::DevicePropType& prop,
                                              const api::StreamType& stream,
                                              StackArray<ConstDeviceView<double, 1>, 1> loc,
                                              IndexArray<1> grid_size, StackArray<double, 1> offset,
                                              StackArray<double, 1> scaling_factor,
                                              DeviceView<PartitionGroup, 1> partition,
                                              DeviceView<Point<double, 1>, 1> points) -> void;

template auto rescale_and_permut_t3<double, 2>(const api::DevicePropType& prop,
                                              const api::StreamType& stream,
                                              StackArray<ConstDeviceView<double, 1>, 2> loc,
                                              IndexArray<2> grid_size, StackArray<double, 2> offset,
                                              StackArray<double, 2> scaling_factor,
                                              DeviceView<PartitionGroup, 2> partition,
                                              DeviceView<Point<double, 2>, 1> points) -> void;

template auto rescale_and_permut_t3<double, 3>(const api::DevicePropType& prop,
                                              const api::StreamType& stream,
                                              StackArray<ConstDeviceView<double, 1>, 3> loc,
                                              IndexArray<3> grid_size, StackArray<double, 3> offset,
                                              StackArray<double, 3> scaling_factor,
                                              DeviceView<PartitionGroup, 3> partition,
                                              DeviceView<Point<double, 3>, 1> points) -> void;

}  // namespace gpu
}  // namespace neonufft
