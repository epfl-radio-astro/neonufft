#include "neonufft/config.h"
//---
#include "neonufft/gpu/kernels/upsample_kernel.hpp"

#include <algorithm>
#include <array>

#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/util/kernel_launch_grid.hpp"
#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/enums.h"
#include "neonufft/gpu/types.hpp"
#include "neonufft/util/stack_array.hpp"

namespace neonufft {
namespace gpu {

__device__ __forceinline__ static float calc_cos(float value) {
  return cosf(value);
}

__device__ __forceinline__ static double calc_cos(double value) {
  return cos(value);
}

__device__ __forceinline__ static float calc_sin(float value) {
  return sinf(value);
}

__device__ __forceinline__ static double calc_sin(double value) {
  return sin(value);
}

template <typename T, IntType DIM, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    prephase_kernel(T sign, StackArray<ConstDeviceView<T, 1>, DIM> input_points,
                    StackArray<T, DIM> out_offset, DeviceView<ComplexType<T>, 1> prephase) {
  for (IntType idx = threadIdx.x + BLOCK_SIZE * blockIdx.x; idx < prephase.size();
       idx += BLOCK_SIZE * gridDim.x) {
    auto phase = input_points[0][idx] * out_offset[0];
    if constexpr (DIM >= 2) {
      phase += input_points[1][idx] * out_offset[1];
    }
    if constexpr (DIM >= 3) {
      phase += input_points[2][idx] * out_offset[2];
    }
    prephase[idx] = ComplexType<T>{calc_cos(phase), sign * calc_sin(phase)};
  }
}

template <typename T, IntType DIM>
auto compute_prephase(const api::DevicePropType& prop, const api::StreamType& stream, int sign,
                      StackArray<ConstDeviceView<T, 1>, DIM> input_points,
                      StackArray<T, DIM> out_offset, DeviceView<ComplexType<T>, 1> prephase)
    -> void {
  constexpr int block_size = 256;

  const dim3 block_dim(block_size, 1, 1);
  const auto grid_dim = kernel_launch_grid(prop, {prephase.size(), 1, 1}, block_dim);

  api::launch_kernel(prephase_kernel<T, DIM, block_size>, grid_dim, block_dim, 0, stream, sign,
                     input_points, out_offset, prephase);
}

template auto compute_prephase<float, 1>(const api::DevicePropType& prop,
                                         const api::StreamType& stream, int sign,
                                         StackArray<ConstDeviceView<float, 1>, 1> input_points,
                                         StackArray<float, 1> out_offset,
                                         DeviceView<ComplexType<float>, 1> prephase) -> void;

template auto compute_prephase<float, 2>(const api::DevicePropType& prop,
                                         const api::StreamType& stream, int sign,
                                         StackArray<ConstDeviceView<float, 1>, 2> input_points,
                                         StackArray<float, 2> out_offset,
                                         DeviceView<ComplexType<float>, 1> prephase) -> void;

template auto compute_prephase<float, 3>(const api::DevicePropType& prop,
                                         const api::StreamType& stream, int sign,
                                         StackArray<ConstDeviceView<float, 1>, 3> input_points,
                                         StackArray<float, 3> out_offset,
                                         DeviceView<ComplexType<float>, 1> prephase) -> void;

template auto compute_prephase<double, 1>(const api::DevicePropType& prop,
                                         const api::StreamType& stream, int sign,
                                         StackArray<ConstDeviceView<double, 1>, 1> input_points,
                                         StackArray<double, 1> out_offset,
                                         DeviceView<ComplexType<double>, 1> prephase) -> void;

template auto compute_prephase<double, 2>(const api::DevicePropType& prop,
                                         const api::StreamType& stream, int sign,
                                         StackArray<ConstDeviceView<double, 1>, 2> input_points,
                                         StackArray<double, 2> out_offset,
                                         DeviceView<ComplexType<double>, 1> prephase) -> void;

template auto compute_prephase<double, 3>(const api::DevicePropType& prop,
                                          const api::StreamType& stream, int sign,
                                          StackArray<ConstDeviceView<double, 1>, 3> input_points,
                                          StackArray<double, 3> out_offset,
                                          DeviceView<ComplexType<double>, 1> prephase) -> void;

}  // namespace gpu
}  // namespace neonufft
