#include "neonufft//config.h"
//---
#include <algorithm>

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/gpu/kernels/es_kernel_eval.cuh"
#include "neonufft/gpu/kernels/spreading_kernel.hpp"
#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/gpu/util/kernel_launch_grid.hpp"
#include "neonufft/gpu/util/partition_group.hpp"
#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/util/point.hpp"

namespace neonufft {
namespace gpu {

template <typename KER, typename T, int N_SPREAD>
__device__ static void spread_points(const KER& kernel, IndexArray<1> thread_grid_idx,
                                     ConstDeviceView<Point<T, 1>, 1> points,
                                     ConstDeviceView<ComplexType<T>, 1> input,
                                     ConstDeviceView<ComplexType<T>, 1> prephase_optional,
                                     IndexArray<1> grid_shape, T* __restrict__ ker,
                                     ComplexType<T>& sum) {
  assert(blockDim.x == PartitionGroup::width);

  constexpr T half_width = T(N_SPREAD) / T(2);  // half spread width
  const auto thread_grid_idx_x = thread_grid_idx[0];

  for (int idx_p = 0; idx_p < points.size(); ++idx_p) {
    const auto point = points[idx_p];

    const T loc_x = point.coord[0] * grid_shape[0];  // px in [0, 1]
    const T idx_init_ceil_x = ceil(loc_x - half_width);
    const IntType idx_init_x = IntType(idx_init_ceil_x);  // fine grid start index (-
    const T x = idx_init_ceil_x - loc_x;                  // x1 in [-w/2,-w/2+1], up to rounding

    // precompute kernel
    if (threadIdx.x < N_SPREAD) {
      ker[threadIdx.x] = kernel.eval_scalar(x + threadIdx.x);
    }
    __syncthreads();

    if (thread_grid_idx_x >= idx_init_x && thread_grid_idx_x < idx_init_x + N_SPREAD) {
      const auto ker_value_x = ker[thread_grid_idx_x - idx_init_x];
      auto in_value = input[point.index];
      if (prephase_optional.size()) {
        const auto pre = prephase_optional[point.index];
        in_value.x = in_value.x * pre.x - in_value.y * pre.y;
        in_value.y = in_value.x * pre.y + in_value.y * pre.x;
      }

      sum.x += ker_value_x * in_value.x;
      sum.y += ker_value_x * in_value.y;
    }
  }
}

template <typename KER, typename T, int N_SPREAD>
__device__ static void spread_points(const KER& kernel, IndexArray<2> thread_grid_idx,
                                     ConstDeviceView<Point<T, 2>, 1> points,
                                     ConstDeviceView<ComplexType<T>, 1> input,
                                     ConstDeviceView<ComplexType<T>, 1> prephase_optional,
                                     IndexArray<2> grid_shape, T* __restrict__ ker,
                                     ComplexType<T>& sum) {
  assert(blockDim.x == PartitionGroup::width);

  constexpr T half_width = T(N_SPREAD) / T(2);  // half spread width

  const int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

  for (int idx_p = 0; idx_p < points.size(); ++idx_p) {
    const auto point = points[idx_p];

    const T loc_x = point.coord[0] * grid_shape[0];  // px in [0, 1]
    const T idx_init_ceil_x = ceil(loc_x - half_width);
    const IntType idx_init_x = IntType(idx_init_ceil_x);  // fine grid start index (-
    const T x = idx_init_ceil_x - loc_x;                  // x1 in [-w/2,-w/2+1], up to rounding

    const T loc_y = point.coord[1] * grid_shape[1];
    const T idx_init_ceil_y = ceil(loc_y - half_width);
    const IntType idx_init_y = IntType(idx_init_ceil_y);
    const T y = idx_init_ceil_y - loc_y;

    // precompute kernel
    // expensive kernel evaluation, ideally computed in single warp
    if (thread_idx < 2 * N_SPREAD) {
      const T ker_arg = thread_idx < N_SPREAD ? x + thread_idx : y + (thread_idx - N_SPREAD);
      ker[thread_idx] = kernel.eval_scalar(ker_arg);
    }
    __syncthreads();

    if (thread_grid_idx[0] >= idx_init_x && thread_grid_idx[0] < idx_init_x + N_SPREAD &&
        thread_grid_idx[1] >= idx_init_y && thread_grid_idx[1] < idx_init_y + N_SPREAD) {
      const auto ker_value_x = ker[thread_grid_idx[0] - idx_init_x];
      const auto ker_value_y = ker[thread_grid_idx[1] - idx_init_y + N_SPREAD];
      auto in_value = input[point.index];
      if (prephase_optional.size()) {
        const auto pre = prephase_optional[point.index];
        in_value.x = in_value.x * pre.x - in_value.y * pre.y;
        in_value.y = in_value.x * pre.y + in_value.y * pre.x;
      }

      const auto ker_value = ker_value_x * ker_value_y;
      sum.x += ker_value * in_value.x;
      sum.y += ker_value * in_value.y;
    }
  }
}

template <typename KER, typename T, IntType DIM, int N_SPREAD>
__device__ static void spread_partition_group_1d(
    const KER& kernel, IntType idx_block_part_x, IndexArray<DIM> thread_grid_idx,
    ConstDeviceView<PartitionGroup, 1> partition, ConstDeviceView<Point<T, DIM>, 1> points,
    ConstDeviceView<ComplexType<T>, 1> input, ConstDeviceView<ComplexType<T>, 1> prephase_optional,
    IndexArray<DIM> grid_shape, T* __restrict__ ker, ComplexType<T>& sum) {

  // check from left to right +-1
  for (IntType idx_part_x = idx_block_part_x > 0 ? idx_block_part_x - 1 : 0;
       idx_part_x <= idx_block_part_x + 1 && idx_part_x < partition.shape(0); ++idx_part_x) {
    const auto part = partition[idx_part_x];
    spread_points<KER, T, N_SPREAD>(kernel, thread_grid_idx, points.sub_view(part.begin, part.size),
                                    input, prephase_optional, grid_shape, ker, sum);
  }

  // periodic wrap around with shifted grid index. We check the two last partition cells, because
  // the the last one might not be fully filled up and points in the second last may still be
  // within N_SPREAD distance.
  if (idx_block_part_x == 0) {
    auto part = partition[partition.shape(0) - 1];
    auto thread_grid_idx_shifted = thread_grid_idx;
    thread_grid_idx_shifted[0] += grid_shape[0];
    spread_points<KER, T, N_SPREAD>(kernel, thread_grid_idx_shifted,
                                    points.sub_view(part.begin, part.size), input,
                                    prephase_optional, grid_shape, ker, sum);
    if (partition.shape(0) > 1) {
      part = partition[partition.shape(0) - 2];
      spread_points<KER, T, N_SPREAD>(kernel, thread_grid_idx_shifted,
                                      points.sub_view(part.begin, part.size), input,
                                      prephase_optional, grid_shape, ker, sum);
    }
  }

  if (idx_block_part_x >= partition.shape(0) - 2) {
    auto part = partition[0];
    auto thread_grid_idx_shifted = thread_grid_idx;
    thread_grid_idx_shifted[0] -= grid_shape[0];
    spread_points<KER, T, N_SPREAD>(kernel, thread_grid_idx_shifted,
                                    points.sub_view(part.begin, part.size), input,
                                    prephase_optional, grid_shape, ker, sum);
  }
}

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    spread_1d_kernel(KER kernel, ConstDeviceView<PartitionGroup, 1> partition,
                     ConstDeviceView<Point<T, 1>, 1> points,
                     ConstDeviceView<ComplexType<T>, 1> input,
                     ConstDeviceView<ComplexType<T>, 1> prephase_optional,
                     DeviceView<ComplexType<T>, 1> grid) {
  static_assert(BLOCK_SIZE == PartitionGroup::width);
  __shared__ T ker[N_SPREAD];

  for (IntType idx_block_part_x = blockIdx.x; idx_block_part_x < partition.shape(0);
       idx_block_part_x += gridDim.x) {
    ComplexType<T> sum{0, 0};

    const IntType thread_grid_idx_x = threadIdx.x + idx_block_part_x * PartitionGroup::width;
    spread_partition_group_1d<KER, T, 1, N_SPREAD>(kernel, idx_block_part_x, thread_grid_idx_x,
                                                   partition, points, input, prephase_optional,
                                                   grid.shape(), ker, sum);

    if (thread_grid_idx_x < grid.shape(0) && (sum.x || sum.y)) {
      auto value = grid[thread_grid_idx_x];
      value.x += sum.x;
      value.y += sum.y;
      grid[thread_grid_idx_x] = value;
    }
  }
}

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE)
    spread_2d_kernel(KER kernel, ConstDeviceView<PartitionGroup, 2> partition,
                     ConstDeviceView<Point<T, 2>, 1> points,
                     ConstDeviceView<ComplexType<T>, 1> input,
                     ConstDeviceView<ComplexType<T>, 1> prephase_optional,
                     DeviceView<ComplexType<T>, 2> grid) {
  static_assert(BLOCK_SIZE == PartitionGroup::width);
  __shared__ T ker[2 * N_SPREAD];

  for (IntType idx_block_part_y = blockIdx.y; idx_block_part_y < partition.shape(1);
       idx_block_part_y += gridDim.y) {
    const IntType thread_grid_idx_y = threadIdx.y + idx_block_part_y * PartitionGroup::width;
    for (IntType idx_block_part_x = blockIdx.x; idx_block_part_x < partition.shape(0);
         idx_block_part_x += gridDim.x) {
      ComplexType<T> sum{0, 0};

      const IntType thread_grid_idx_x = threadIdx.x + idx_block_part_x * PartitionGroup::width;

      // check from left to right +-1
      for (IntType idx_part_y = idx_block_part_y > 0 ? idx_block_part_y - 1 : 0;
           idx_part_y <= idx_block_part_y + 1 && idx_part_y < partition.shape(1); ++idx_part_y) {
        spread_partition_group_1d<KER, T, 2, N_SPREAD>(
            kernel, idx_block_part_x, {thread_grid_idx_x, thread_grid_idx_y},
            partition.slice_view(idx_part_y), points, input, prephase_optional, grid.shape(), ker,
            sum);
      }

      // check around periodic boundary
      if (idx_block_part_y == 0) {
        IndexArray<2> thread_grid_idx_shifted{thread_grid_idx_x, thread_grid_idx_y + grid.shape(1)};
        spread_partition_group_1d<KER, T, 2, N_SPREAD>(
            kernel, idx_block_part_x, thread_grid_idx_shifted,
            partition.slice_view(partition.shape(1) - 1), points, input, prephase_optional,
            grid.shape(), ker, sum);
        if (partition.shape(1) > 1) {
          spread_partition_group_1d<KER, T, 2, N_SPREAD>(
              kernel, idx_block_part_x, thread_grid_idx_shifted,
              partition.slice_view(partition.shape(1) - 2), points, input, prephase_optional,
              grid.shape(), ker, sum);
        }
      }

      if (idx_block_part_y >= partition.shape(1) - 2) {
        IndexArray<2> thread_grid_idx_shifted{thread_grid_idx_x, thread_grid_idx_y - grid.shape(1)};
        spread_partition_group_1d<KER, T, 2, N_SPREAD>(
            kernel, idx_block_part_x, thread_grid_idx_shifted, partition.slice_view(0), points,
            input, prephase_optional, grid.shape(), ker, sum);
      }

      if (thread_grid_idx_x < grid.shape(0) && thread_grid_idx_y < grid.shape(1) &&
          (sum.x || sum.y)) {
        auto value = grid[{thread_grid_idx_x, idx_block_part_y}];
        value.x += sum.x;
        value.y += sum.y;
        grid[{thread_grid_idx_x, idx_block_part_y}] = value;
      }
    }
  }
}

template <typename T, IntType DIM, int N_SPREAD, int BLOCK_SIZE>
auto spread_dispatch(const api::DevicePropType& prop, const api::StreamType& stream,
                     const KernelParameters<T>& param,
                     ConstDeviceView<PartitionGroup, DIM> partition,
                     ConstDeviceView<Point<T, DIM>, 1> points,
                     ConstDeviceView<ComplexType<T>, 1> input,
                     ConstDeviceView<ComplexType<T>, 1> prephase_optional,
                     std::array<IntType, DIM> grid_size, DeviceView<ComplexType<T>, DIM> grid)
    -> void {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  if (param.n_spread == N_SPREAD) {
    EsKernelDirect<T, N_SPREAD> kernel{param.es_beta};

    if constexpr (DIM == 1) {
      const dim3 block_dim(BLOCK_SIZE, 1, 1);
      const dim3 grid_dim(std::min<int>(partition.shape(0), prop.maxGridSize[0]), 1, 1);
      api::launch_kernel(spread_1d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE>, grid_dim,
                         block_dim, 0, stream, kernel, partition, points, input, prephase_optional,
                         grid);
    } else if constexpr (DIM == 2) {
      const dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
      const dim3 grid_dim(std::min<int>(partition.shape(0), prop.maxGridSize[0]),
                          std::min<int>(partition.shape(1), prop.maxGridSize[1]), 1);
      api::launch_kernel(spread_2d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE>, grid_dim,
                         block_dim, 0, stream, kernel, partition, points, input, prephase_optional,
                         grid);
    } else {
    }
  } else {
    if constexpr (N_SPREAD > 2) {
      spread_dispatch<T, DIM, N_SPREAD - 1, BLOCK_SIZE>(prop, stream, param, partition, points,
                                                        input, prephase_optional, grid_size, grid);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}

template <typename T, IntType DIM>
void spread(const api::DevicePropType& prop, const api::StreamType& stream,
            const KernelParameters<T>& param, ConstDeviceView<PartitionGroup, DIM> partition,
            ConstDeviceView<Point<T, DIM>, 1> points, ConstDeviceView<ComplexType<T>, 1> input,
            ConstDeviceView<ComplexType<T>, 1> prephase_optional,
            std::array<IntType, DIM> grid_size, DeviceView<ComplexType<T>, DIM> grid) {
  constexpr int block_size = PartitionGroup::width;
  spread_dispatch<T, DIM, 16, block_size>(prop, stream, param, partition, points, input,
                                          prephase_optional, grid_size, grid);
}

template void spread<float, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<float>& param,
                               ConstDeviceView<PartitionGroup, 1> partition,
                               ConstDeviceView<Point<float, 1>, 1> points,
                               ConstDeviceView<ComplexType<float>, 1> input,
                               ConstDeviceView<ComplexType<float>, 1> prephase_optional,
                               std::array<IntType, 1> grid_size,
                               DeviceView<ComplexType<float>, 1> grid);

template void spread<float, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<float>& param,
                               ConstDeviceView<PartitionGroup, 2> partition,
                               ConstDeviceView<Point<float, 2>, 1> points,
                               ConstDeviceView<ComplexType<float>, 1> input,
                               ConstDeviceView<ComplexType<float>, 1> prephase_optional,
                               std::array<IntType, 2> grid_size,
                               DeviceView<ComplexType<float>, 2> grid);

template void spread<float, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<float>& param,
                               ConstDeviceView<PartitionGroup, 3> partition,
                               ConstDeviceView<Point<float, 3>, 1> points,
                               ConstDeviceView<ComplexType<float>, 1> input,
                               ConstDeviceView<ComplexType<float>, 1> prephase_optional,
                               std::array<IntType, 3> grid_size,
                               DeviceView<ComplexType<float>, 3> grid);

template void spread<double, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                                const KernelParameters<double>& param,
                                ConstDeviceView<PartitionGroup, 1> partition,
                                ConstDeviceView<Point<double, 1>, 1> points,
                                ConstDeviceView<ComplexType<double>, 1> input,
                                ConstDeviceView<ComplexType<double>, 1> prephase_optional,
                                std::array<IntType, 1> grid_size,
                                DeviceView<ComplexType<double>, 1> grid);

template void spread<double, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                                const KernelParameters<double>& param,
                                ConstDeviceView<PartitionGroup, 2> partition,
                                ConstDeviceView<Point<double, 2>, 1> points,
                                ConstDeviceView<ComplexType<double>, 1> input,
                                ConstDeviceView<ComplexType<double>, 1> prephase_optional,
                                std::array<IntType, 2> grid_size,
                                DeviceView<ComplexType<double>, 2> grid);

template void spread<double, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                const KernelParameters<double>& param,
                                ConstDeviceView<PartitionGroup, 3> partition,
                                ConstDeviceView<Point<double, 3>, 1> points,
                                ConstDeviceView<ComplexType<double>, 1> input,
                                ConstDeviceView<ComplexType<double>, 1> prephase_optional,
                                std::array<IntType, 3> grid_size,
                                DeviceView<ComplexType<double>, 3> grid);

}  // namespace gpu
}  // namespace neonufft
