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

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE>
__device__ static void spread_partition_group_1d(
    const KER& kernel, IntType thread_grid_idx_x, ConstDeviceView<Point<T, 1>, 1> points,
    ConstDeviceView<ComplexType<T>, 1> input, ConstDeviceView<ComplexType<T>, 1> prephase_optional,
    IntType grid_size_x, IntType padding_x, T* __restrict__ ker_x, ComplexType<T>& sum) {
  static_assert(BLOCK_SIZE == PartitionGroup::width);
  constexpr T half_width = T(N_SPREAD) / T(2);  // half spread width

  for (int idx_p = 0; idx_p < points.size(); ++idx_p) {
    const auto point = points[idx_p];
    const T loc = point.coord[0] * grid_size_x;  // px in [0, 1]
    const T idx_init_ceil = ceil(loc - half_width);
    IntType idx_init = IntType(idx_init_ceil);  // fine padded_grid start index (-
    T x = idx_init_ceil - loc;                  // x1 in [-w/2,-w/2+1], up to rounding

    // precompute kernel
    if (threadIdx.x < N_SPREAD) {
      ker_x[threadIdx.x] = kernel.eval_scalar(x + threadIdx.x);
    }
    __syncthreads();

    // offset into padded grid
    idx_init += padding_x;

    if (thread_grid_idx_x >= idx_init && thread_grid_idx_x < idx_init + N_SPREAD) {
      const auto ker_value_x = ker_x[thread_grid_idx_x - idx_init];
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

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    spread_1d_kernel(KER kernel, ConstDeviceView<PartitionGroup, 1> partition,
                     ConstDeviceView<Point<T, 1>, 1> points,
                     ConstDeviceView<ComplexType<T>, 1> input,
                     ConstDeviceView<ComplexType<T>, 1> prephase_optional, IntType grid_size_x,
                     DeviceView<ComplexType<T>, 1> padded_grid) {
  static_assert(BLOCK_SIZE == PartitionGroup::width);
  __shared__ T ker_x[N_SPREAD];

  const IntType padding_x = (padded_grid.shape(0) - grid_size_x) / 2;

  for (IntType idx_block_part_x = blockIdx.x; idx_block_part_x < partition.shape(0);
       idx_block_part_x += gridDim.x) {
    ComplexType<T> sum{0, 0};
    const IntType thread_grid_idx_x = threadIdx.x + idx_block_part_x * PartitionGroup::width;

    // check from left to right +-1
    for (IntType idx_part_x = idx_block_part_x > 0 ? idx_block_part_x - 1 : 0;
         idx_part_x <= idx_block_part_x + 1 && idx_part_x < partition.shape(0); ++idx_part_x) {
      const auto part = partition[idx_part_x];
      spread_partition_group_1d<KER, T, N_SPREAD, BLOCK_SIZE>(
          kernel, thread_grid_idx_x, points.sub_view(part.begin, part.size), input,
          prephase_optional, grid_size_x, padding_x, ker_x, sum);
    }

    // periodic wrap around with shifted grid index. We check the two last partition cells, because
    // the the last one might not be fully filled up and points in the second last may still be
    // within N_SPREAD distance.
    if (idx_block_part_x == 0) {
      auto part = partition[partition.shape(0) - 1];
      spread_partition_group_1d<KER, T, N_SPREAD, BLOCK_SIZE>(
          kernel, thread_grid_idx_x + grid_size_x, points.sub_view(part.begin, part.size), input,
          prephase_optional, grid_size_x, padding_x, ker_x, sum);
      if (partition.shape(0) > 1) {
        part = partition[partition.shape(0) - 2];
        spread_partition_group_1d<KER, T, N_SPREAD, BLOCK_SIZE>(
            kernel, thread_grid_idx_x + grid_size_x, points.sub_view(part.begin, part.size), input,
            prephase_optional, grid_size_x, padding_x, ker_x, sum);
      }
    }

    if (idx_block_part_x >= partition.shape(0) - 2) {
      auto part = partition[0];
      spread_partition_group_1d<KER, T, N_SPREAD, BLOCK_SIZE>(
          kernel, thread_grid_idx_x - grid_size_x, points.sub_view(part.begin, part.size), input,
          prephase_optional, grid_size_x, padding_x, ker_x, sum);
    }

    if (thread_grid_idx_x < padded_grid.shape(0) && (sum.x || sum.y)) {
      auto value = padded_grid[thread_grid_idx_x];
      value.x += sum.x;
      value.y += sum.y;
      padded_grid[thread_grid_idx_x] = value;
    }
  }
}

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    spread_2d_kernel(KER kernel, ConstDeviceView<Point<T, 1>, 1> points,
                     ConstDeviceView<ComplexType<T>, 1> input,
                     ConstDeviceView<ComplexType<T>, 1> prephase_optional, IntType grid_size_x,
                     IntType grid_size_y, DeviceView<ComplexType<T>, 1> padded_grid) {
  static_assert(BLOCK_SIZE == PartitionGroup::width * PartitionGroup::width);
}

template <typename T, IntType DIM, int N_SPREAD, int BLOCK_SIZE>
auto spread_dispatch(const api::DevicePropType& prop, const api::StreamType& stream,
                     const KernelParameters<T>& param,
                     ConstDeviceView<PartitionGroup, DIM> partition,
                     ConstDeviceView<Point<T, DIM>, 1> points,
                     ConstDeviceView<ComplexType<T>, 1> input,
                     ConstDeviceView<ComplexType<T>, 1> prephase_optional,
                     std::array<IntType, DIM> grid_size,
                     DeviceView<ComplexType<T>, DIM> padded_grid) -> void {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  if (param.n_spread == N_SPREAD) {
    EsKernelDirect<T, N_SPREAD> kernel{param.es_beta};
    const dim3 block_dim(std::min<int>(BLOCK_SIZE, prop.maxThreadsDim[0]), 1, 1);
    const dim3 grid_dim(std::min<int>(partition.shape(0), prop.maxGridSize[0]), 1, 1);

    if constexpr (DIM == 1) {
      api::launch_kernel(spread_1d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE>, grid_dim,
                         block_dim, 0, stream, kernel, partition, points, input, prephase_optional,
                         grid_size[0], padded_grid);
    } else if constexpr (DIM == 2) {
      // api::launch_kernel(
      //     interpolation_2d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE, WARP_SIZE>, grid_dim,
      //     block_dim, 0, stream, kernel, points, grid, out);
    } else {
      // api::launch_kernel(
      //     interpolation_3d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE, WARP_SIZE>, grid_dim,
      //     block_dim, 0, stream, kernel, points, grid, out);
    }
  } else {
    if constexpr (N_SPREAD > 2) {
      spread_dispatch<T, DIM, N_SPREAD - 1, BLOCK_SIZE>(
          prop, stream, param, partition, points, input, prephase_optional, grid_size, padded_grid);
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
            std::array<IntType, DIM> grid_size, DeviceView<ComplexType<T>, DIM> padded_grid) {
  constexpr int block_size = PartitionGroup::width;
  spread_dispatch<T, DIM, 16, block_size>(prop, stream, param, partition, points, input,
                                          prephase_optional, grid_size, padded_grid);
}

template void spread<float, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<float>& param,
                               ConstDeviceView<PartitionGroup, 1> partition,
                               ConstDeviceView<Point<float, 1>, 1> points,
                               ConstDeviceView<ComplexType<float>, 1> input,
                               ConstDeviceView<ComplexType<float>, 1> prephase_optional,
                               std::array<IntType, 1> grid_size,
                               DeviceView<ComplexType<float>, 1> padded_grid);

template void spread<float, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<float>& param,
                               ConstDeviceView<PartitionGroup, 2> partition,
                               ConstDeviceView<Point<float, 2>, 1> points,
                               ConstDeviceView<ComplexType<float>, 1> input,
                               ConstDeviceView<ComplexType<float>, 1> prephase_optional,
                               std::array<IntType, 2> grid_size,
                               DeviceView<ComplexType<float>, 2> padded_grid);

template void spread<float, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<float>& param,
                               ConstDeviceView<PartitionGroup, 3> partition,
                               ConstDeviceView<Point<float, 3>, 1> points,
                               ConstDeviceView<ComplexType<float>, 1> input,
                               ConstDeviceView<ComplexType<float>, 1> prephase_optional,
                               std::array<IntType, 3> grid_size,
                               DeviceView<ComplexType<float>, 3> padded_grid);

template void spread<double, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<double>& param,
                               ConstDeviceView<PartitionGroup, 1> partition,
                               ConstDeviceView<Point<double, 1>, 1> points,
                               ConstDeviceView<ComplexType<double>, 1> input,
                               ConstDeviceView<ComplexType<double>, 1> prephase_optional,
                               std::array<IntType, 1> grid_size,
                               DeviceView<ComplexType<double>, 1> padded_grid);

template void spread<double, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                               const KernelParameters<double>& param,
                               ConstDeviceView<PartitionGroup, 2> partition,
                               ConstDeviceView<Point<double, 2>, 1> points,
                               ConstDeviceView<ComplexType<double>, 1> input,
                               ConstDeviceView<ComplexType<double>, 1> prephase_optional,
                               std::array<IntType, 2> grid_size,
                               DeviceView<ComplexType<double>, 2> padded_grid);

template void spread<double, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                const KernelParameters<double>& param,
                                ConstDeviceView<PartitionGroup, 3> partition,
                                ConstDeviceView<Point<double, 3>, 1> points,
                                ConstDeviceView<ComplexType<double>, 1> input,
                                ConstDeviceView<ComplexType<double>, 1> prephase_optional,
                                std::array<IntType, 3> grid_size,
                                DeviceView<ComplexType<double>, 3> padded_grid);

}  // namespace gpu
}  // namespace neonufft
