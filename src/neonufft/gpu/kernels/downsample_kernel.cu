#include "neonufft/config.h"
//---
#include "neonufft/gpu/kernels/downsample_kernel.hpp"

#include <algorithm>
#include <array>

#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/util/kernel_launch_grid.hpp"
#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/enums.h"
#include "neonufft/gpu/types.hpp"

namespace neonufft {
namespace gpu {

template <typename T, int BLOCK_SIZE>
__device__ static void downsample_1d_inner_kernel(NeonufftModeOrder order,
                                                  ConstDeviceView<ComplexType<T>, 1> large_grid,
                                                  T prefac, ConstDeviceView<T, 1> ker_x,
                                                  DeviceView<ComplexType<T>, 1> small_grid) {
  const auto n_small = small_grid.shape(0);
  const IntType n_negative = n_small / 2;               // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;  // [0, N/2 -1]

  const IntType thread_start = threadIdx.x + blockIdx.x * BLOCK_SIZE;

  IntType idx_out_non_negative = n_negative + thread_start;
  IntType idx_out_negative = thread_start;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_out_non_negative = thread_start;
    idx_out_negative = n_non_negative + thread_start;
  }

  const IntType offset_negative = large_grid.shape(0) - n_negative;
  const IntType block_step = gridDim.x * BLOCK_SIZE;

  for (IntType idx = thread_start; idx < n_non_negative;
       idx += block_step, idx_out_non_negative += block_step) {
    auto value = large_grid[idx];
    auto ker_value = prefac * ker_x[idx];
    value.x *= ker_value;
    value.y *= ker_value;
    small_grid[idx_out_non_negative] = value;
  }

  for (IntType idx = thread_start; idx < n_negative;
       idx += block_step, idx_out_negative += block_step) {
    auto value = large_grid[idx + offset_negative];
    auto ker_value = prefac * ker_x[n_negative - idx];
    value.x *= ker_value;
    value.y *= ker_value;
    small_grid[idx_out_negative] = value;
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    downsample_1d_kernel(NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, 1> large_grid,
                         ConstDeviceView<T, 1> ker_x, DeviceView<ComplexType<T>, 1> small_grid) {
  downsample_1d_inner_kernel<T, BLOCK_SIZE>(order, large_grid, 1, ker_x, small_grid);
}

template <typename T, int BLOCK_SIZE>
__device__ static void downsample_2d_inner_kernel(NeonufftModeOrder order,
                                                  ConstDeviceView<ComplexType<T>, 2> large_grid,
                                                  T prefac, ConstDeviceView<T, 1> ker_x,
                                                  ConstDeviceView<T, 1> ker_y,
                                                  DeviceView<ComplexType<T>, 2> small_grid) {
  const auto n_small = small_grid.shape(1);

  const IntType n_negative = n_small / 2;               // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;  // [0, N/2 -1]

  const IntType block_start = blockIdx.y;

  IntType idx_out_non_negative = n_negative + block_start;
  IntType idx_out_negative = block_start;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_out_non_negative = block_start;
    idx_out_negative = n_non_negative + block_start;
  }

  const IntType padding = large_grid.shape(1) - small_grid.shape(1);

  for (IntType idx = block_start; idx < n_non_negative;
       idx += gridDim.y, idx_out_non_negative += gridDim.y) {
    const auto ker_value = prefac * ker_y[idx];
    downsample_1d_inner_kernel<T, BLOCK_SIZE>(order, large_grid.slice_view(idx), ker_value, ker_x,
                                              small_grid.slice_view(idx_out_non_negative));
  }

  for (IntType idx = block_start; idx < n_negative;
       idx += gridDim.y, idx_out_negative += gridDim.y) {
    const auto ker_value = prefac * ker_y[n_negative - idx];

    downsample_1d_inner_kernel<T, BLOCK_SIZE>(
        order, large_grid.slice_view(n_non_negative + idx + padding), ker_value, ker_x,
        small_grid.slice_view(idx_out_negative));
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    downsample_2d_kernel(NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, 2> large_grid,
                         ConstDeviceView<T, 1> ker_x, ConstDeviceView<T, 1> ker_y,
                         DeviceView<ComplexType<T>, 2> small_grid) {
  downsample_2d_inner_kernel<T, BLOCK_SIZE>(order, large_grid, 1, ker_x, ker_y, small_grid);
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    downsample_3d_kernel(NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, 3> large_grid,
                         ConstDeviceView<T, 1> ker_x, ConstDeviceView<T, 1> ker_y,
                         ConstDeviceView<T, 1> ker_z, DeviceView<ComplexType<T>, 3> small_grid) {
  const auto n_small = small_grid.shape(2);

  const IntType n_negative = n_small / 2;               // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;  // [0, N/2 -1]

  const IntType block_start = blockIdx.z;

  IntType idx_out_non_negative = n_negative + block_start;
  IntType idx_out_negative = block_start;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_out_non_negative = block_start;
    idx_out_negative = n_non_negative + block_start;
  }

  const IntType padding = large_grid.shape(2) - small_grid.shape(2);

  for (IntType idx = block_start; idx < n_non_negative;
       idx += gridDim.z, idx_out_non_negative += gridDim.z) {
    const auto ker_value = ker_z[idx];
    downsample_2d_inner_kernel<T, BLOCK_SIZE>(order, large_grid.slice_view(idx), ker_value, ker_x,
                                              ker_y, small_grid.slice_view(idx_out_non_negative));
  }

  for (IntType idx = block_start; idx < n_negative;
       idx += gridDim.z, idx_out_negative += gridDim.z) {
    const auto ker_value = ker_z[n_negative - idx];

    downsample_2d_inner_kernel<T, BLOCK_SIZE>(
        order, large_grid.slice_view(n_non_negative + idx + padding), ker_value, ker_x, ker_y,
        small_grid.slice_view(idx_out_negative));
  }
}

template <typename T, IntType DIM>
auto downsample(const api::DevicePropType& prop, const api::StreamType& stream,
                NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, DIM> large_grid,
                std::array<ConstDeviceView<T, 1>, DIM> ker,
                DeviceView<ComplexType<T>, DIM> small_grid) -> void {
  constexpr int block_size = 128;

  const IntType n_negative_x = small_grid.shape(0) / 2;                 // [-N/2, -1]
  const IntType n_non_negative_x = small_grid.shape(0) - n_negative_x;  // [0, N/2 -1]

  if constexpr (DIM == 1) {
    const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(prop, {n_non_negative_x, 1, 1}, block);
    api::launch_kernel(downsample_1d_kernel<T, block_size>, grid, block, 0, stream, order,
                       large_grid, ker[0], small_grid);

  } else if constexpr (DIM == 2) {
    const IntType n_negative_y = small_grid.shape(1) / 2;                 // [-N/2, -1]
    const IntType n_non_negative_y = small_grid.shape(1) - n_negative_y;  // [0, N/2 -1]
    const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(prop, {n_non_negative_x, n_non_negative_y, 1}, block);
    api::launch_kernel(downsample_2d_kernel<T, block_size>, grid, block, 0, stream, order,
                       large_grid, ker[0], ker[1], small_grid);

  } else if constexpr (DIM == 3) {
    const IntType n_negative_y = small_grid.shape(1) / 2;                 // [-N/2, -1]
    const IntType n_non_negative_y = small_grid.shape(1) - n_negative_y;  // [0, N/2 -1]
    const IntType n_negative_z = small_grid.shape(2) / 2;                 // [-N/2, -1]
    const IntType n_non_negative_z = small_grid.shape(2) - n_negative_z;  // [0, N/2 -1]
                                                                          //
    const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
    const auto grid =
        kernel_launch_grid(prop, {n_non_negative_x, n_non_negative_y, n_non_negative_z}, block);
    api::launch_kernel(downsample_3d_kernel<T, block_size>, grid, block, 0, stream, order,
                       large_grid, ker[0], ker[1], ker[2], small_grid);
  } else {
    throw InternalError("invalid dimension");
  }
}

template auto downsample<float, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                                   NeonufftModeOrder order,
                                   ConstDeviceView<ComplexType<float>, 1> large_grid,
                                   std::array<ConstDeviceView<float, 1>, 1> ker,
                                   DeviceView<ComplexType<float>, 1> small_grid) -> void;

template auto downsample<float, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                                   NeonufftModeOrder order,
                                   ConstDeviceView<ComplexType<float>, 2> large_grid,
                                   std::array<ConstDeviceView<float, 1>, 2> ker,
                                   DeviceView<ComplexType<float>, 2> small_grid) -> void;

template auto downsample<float, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                   NeonufftModeOrder order,
                                   ConstDeviceView<ComplexType<float>, 3> large_grid,
                                   std::array<ConstDeviceView<float, 1>, 3> ker,
                                   DeviceView<ComplexType<float>, 3> small_grid) -> void;

template auto downsample<double, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                                    NeonufftModeOrder order,
                                    ConstDeviceView<ComplexType<double>, 1> large_grid,
                                    std::array<ConstDeviceView<double, 1>, 1> ker,
                                    DeviceView<ComplexType<double>, 1> small_grid) -> void;

template auto downsample<double, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                                    NeonufftModeOrder order,
                                    ConstDeviceView<ComplexType<double>, 2> large_grid,
                                    std::array<ConstDeviceView<double, 1>, 2> ker,
                                    DeviceView<ComplexType<double>, 2> small_grid) -> void;

template auto downsample<double, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                    NeonufftModeOrder order,
                                    ConstDeviceView<ComplexType<double>, 3> large_grid,
                                    std::array<ConstDeviceView<double, 1>, 3> ker,
                                    DeviceView<ComplexType<double>, 3> small_grid) -> void;

}  // namespace gpu
}  // namespace neonufft
