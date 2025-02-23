#include "neonufft//config.h"
//---
#include "gpu/kernels/upsample_kernel.hpp"

#include <algorithm>
#include <array>

#include "gpu/memory/device_view.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"
#include "neonufft/enums.h"
#include "neonufft/gpu/types.hpp"

namespace neonufft {
namespace gpu {

template <typename T, int BLOCK_SIZE>
__device__ static void upsample_1d_inner_kernel(NeonufftModeOrder order,
                                                ConstDeviceView<ComplexType<T>, 1> small_grid,
                                                T prefac, ConstDeviceView<T, 1> ker_x,
                                                DeviceView<ComplexType<T>, 1> large_grid) {
  const auto n_small = small_grid.shape(0);
  const IntType n_negative = n_small / 2; // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;// [0, N/2 -1]

  const IntType thread_start = threadIdx.x + blockIdx.x * BLOCK_SIZE;

  IntType idx_in_non_negative = n_negative + thread_start;
  IntType idx_in_negative = thread_start;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_in_non_negative = thread_start;
    idx_in_negative = n_non_negative + thread_start;
  }

  const IntType offset_negative = large_grid.shape(0) - n_negative;
  const IntType block_step = gridDim.x * BLOCK_SIZE;

  for (IntType idx = thread_start; idx < n_non_negative;
       idx += block_step, idx_in_non_negative += block_step) {
    auto value = small_grid[idx_in_non_negative];
    auto ker_value = prefac * ker_x[idx];
    value.x *= ker_value;
    value.y *= ker_value;
    large_grid[idx] = value;
  }

  for (IntType idx = thread_start; idx < n_negative;
       idx += block_step, idx_in_negative += block_step) {
    auto value = small_grid[idx_in_negative];
    auto ker_value = prefac * ker_x[n_negative - idx];
    value.x *= ker_value;
    value.y *= ker_value;
    large_grid[idx + offset_negative] = value;
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    upsample_1d_kernel(NeonufftModeOrder order,
                                    ConstDeviceView<ComplexType<T>, 1> small_grid,
                                    ConstDeviceView<T, 1> ker_x,
                                    DeviceView<ComplexType<T>, 1> large_grid) {
  upsample_1d_inner_kernel<T, BLOCK_SIZE>(order, small_grid, 1, ker_x, large_grid);
}

template <typename T, int BLOCK_SIZE>
__device__ static void upsample_2d_inner_kernel(NeonufftModeOrder order,
                                                ConstDeviceView<ComplexType<T>, 2> small_grid,
                                                T prefac,
                                                ConstDeviceView<T, 1> ker_x,
                                                ConstDeviceView<T, 1> ker_y,
                                                DeviceView<ComplexType<T>, 2> large_grid) {
  const auto n_small = small_grid.shape(1);

  const IntType n_negative = n_small / 2; // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;// [0, N/2 -1]

  const IntType block_start = blockIdx.y;

  IntType idx_in_non_negative = n_negative + block_start;
  IntType idx_in_negative = block_start;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_in_non_negative = block_start;
    idx_in_negative = n_non_negative + block_start;
  }

  const IntType padding = large_grid.shape(1) - small_grid.shape(1);

  for (IntType idx = block_start; idx < n_non_negative;
       idx += gridDim.y, idx_in_non_negative += gridDim.y) {
    const auto ker_value = prefac * ker_y[idx];
    upsample_1d_inner_kernel<T, BLOCK_SIZE>(order, small_grid.slice_view(idx_in_non_negative),
                                            ker_value, ker_x, large_grid.slice_view(idx));
  }

  for (IntType idx = block_start; idx < n_negative;
       idx += gridDim.y, idx_in_negative += gridDim.y) {
    const auto ker_value = prefac * ker_y[n_negative - idx];

    upsample_1d_inner_kernel<T, BLOCK_SIZE>(order, small_grid.slice_view(idx_in_negative),
                                            ker_value, ker_x,
                                            large_grid.slice_view(n_non_negative + idx + padding));
  }
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    upsample_2d_kernel(NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, 2> small_grid,
                       ConstDeviceView<T, 1> ker_x, ConstDeviceView<T, 1> ker_y,
                       DeviceView<ComplexType<T>, 2> large_grid) {
  upsample_2d_inner_kernel<T, BLOCK_SIZE>(order, small_grid, 1, ker_x, ker_y, large_grid);
}

template <typename T, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    upsample_3d_kernel(NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, 3> small_grid,
                       ConstDeviceView<T, 1> ker_x, ConstDeviceView<T, 1> ker_y,
                       ConstDeviceView<T, 1> ker_z, DeviceView<ComplexType<T>, 3> large_grid) {
  const auto n_small = small_grid.shape(2);

  const IntType n_negative = n_small / 2;               // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;  // [0, N/2 -1]

  const IntType block_start = blockIdx.z;

  IntType idx_in_non_negative = n_negative + block_start;
  IntType idx_in_negative = block_start;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_in_non_negative = block_start;
    idx_in_negative = n_non_negative + block_start;
  }

  const IntType padding = large_grid.shape(2) - small_grid.shape(2);

  for (IntType idx = block_start; idx < n_non_negative;
       idx += gridDim.z, idx_in_non_negative += gridDim.z) {
    const auto ker_value = ker_z[idx];
    upsample_2d_inner_kernel<T, BLOCK_SIZE>(order, small_grid.slice_view(idx_in_non_negative),
                                            ker_value, ker_x, ker_y, large_grid.slice_view(idx));
  }

  for (IntType idx = block_start; idx < n_negative;
       idx += gridDim.z, idx_in_negative += gridDim.z) {
    const auto ker_value = ker_z[n_negative - idx];

    upsample_2d_inner_kernel<T, BLOCK_SIZE>(order, small_grid.slice_view(idx_in_negative),
                                            ker_value, ker_x, ker_y,
                                            large_grid.slice_view(n_non_negative + idx + padding));
  }
}

template <typename T, IntType DIM>
auto upsample(const api::DevicePropType& prop, const api::StreamType& stream,
              NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, DIM> small_grid,
              std::array<ConstDeviceView<T, 1>, DIM> ker,
              DeviceView<ComplexType<T>, DIM> large_grid) -> void {
  constexpr int block_size = 128;

  const IntType n_negative_x = small_grid.shape(0) / 2;                 // [-N/2, -1]
  const IntType n_non_negative_x = small_grid.shape(0) - n_negative_x;  // [0, N/2 -1]

  if constexpr (DIM == 1) {

    const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(prop, {n_non_negative_x, 1, 1}, block);
    api::launch_kernel(upsample_1d_kernel<T, block_size>, grid, block, 0, stream, order, small_grid,
                       ker[0], large_grid);

  } else if constexpr (DIM == 2){

    const IntType n_negative_y = small_grid.shape(1) / 2;               // [-N/2, -1]
    const IntType n_non_negative_y = small_grid.shape(1) - n_negative_y;  // [0, N/2 -1]
    const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(prop, {n_non_negative_x, n_non_negative_y, 1}, block);
    api::launch_kernel(upsample_2d_kernel<T, block_size>, grid, block, 0, stream, order, small_grid,
                       ker[0], ker[1], large_grid);

  } else if constexpr (DIM == 3) {
    const IntType n_negative_y = small_grid.shape(1) / 2;               // [-N/2, -1]
    const IntType n_non_negative_y = small_grid.shape(1) - n_negative_y;  // [0, N/2 -1]
    const IntType n_negative_z = small_grid.shape(2) / 2;               // [-N/2, -1]
    const IntType n_non_negative_z = small_grid.shape(2) - n_negative_z;  // [0, N/2 -1]
                                                                          //
    const dim3 block(std::min<int>(block_size, prop.maxThreadsDim[0]), 1, 1);
    const auto grid =
        kernel_launch_grid(prop, {n_non_negative_x, n_non_negative_y, n_non_negative_z}, block);
    api::launch_kernel(upsample_3d_kernel<T, block_size>, grid, block, 0, stream, order, small_grid,
                       ker[0], ker[1], ker[2], large_grid);
  } else {
    throw InternalError("invalid dimension");
  }
}

template auto upsample<float, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 NeonufftModeOrder order,
                                 ConstDeviceView<ComplexType<float>, 1> small_grid,
                                 std::array<ConstDeviceView<float, 1>, 1> ker,
                                 DeviceView<ComplexType<float>, 1> large_grid) -> void;

template auto upsample<float, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 NeonufftModeOrder order,
                                 ConstDeviceView<ComplexType<float>, 2> small_grid,
                                 std::array<ConstDeviceView<float, 1>, 2> ker,
                                 DeviceView<ComplexType<float>, 2> large_grid) -> void;

template auto upsample<float, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 NeonufftModeOrder order,
                                 ConstDeviceView<ComplexType<float>, 3> small_grid,
                                 std::array<ConstDeviceView<float, 1>, 3> ker,
                                 DeviceView<ComplexType<float>, 3> large_grid) -> void;

template auto upsample<double, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 NeonufftModeOrder order,
                                 ConstDeviceView<ComplexType<double>, 1> small_grid,
                                 std::array<ConstDeviceView<double, 1>, 1> ker,
                                 DeviceView<ComplexType<double>, 1> large_grid) -> void;

template auto upsample<double, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 NeonufftModeOrder order,
                                 ConstDeviceView<ComplexType<double>, 2> small_grid,
                                 std::array<ConstDeviceView<double, 1>, 2> ker,
                                 DeviceView<ComplexType<double>, 2> large_grid) -> void;

template auto upsample<double, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 NeonufftModeOrder order,
                                 ConstDeviceView<ComplexType<double>, 3> small_grid,
                                 std::array<ConstDeviceView<double, 1>, 3> ker,
                                 DeviceView<ComplexType<double>, 3> large_grid) -> void;

}  // namespace gpu
}  // namespace neonufft
