#include "neonufft/config.h"
//---
#include <algorithm>

#include "neonufft/enums.h"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/gpu/kernels/es_kernel_eval.cuh"
#include "neonufft/gpu/kernels/legendre_rule.cuh"
#include "neonufft/gpu/kernels/postphase_kernel.hpp"
#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/gpu/util/kernel_launch_grid.hpp"
#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/util/stack_array.hpp"

namespace neonufft {
namespace gpu {

static __device__ __forceinline__ float calc_cos(float x) { return cosf(x); }
static __device__ __forceinline__ double calc_cos(double x) { return cos(x); }
static __device__ __forceinline__ float calc_sin(float x) { return sinf(x); }
static __device__ __forceinline__ double calc_sin(double x) { return sin(x); }



template <typename T, IntType DIM, typename KER, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    postphase_kernel(KER kernel, T sign, StackArray<ConstDeviceView<T, 1>, DIM> loc,
                     StackArray<T, DIM> in_offsets, StackArray<T, DIM> out_offsets,
                     StackArray<T, DIM> scaling_factors, DeviceView<ComplexType<T>, 1> postphase) {
  constexpr auto N_SPREAD = KER::n_spread;
  constexpr auto N_QUAD = N_SPREAD + (N_SPREAD % 2);
  constexpr T half_width = T(N_SPREAD) / T(2);

  static_assert(BLOCK_SIZE >= N_QUAD);

  __shared__ T nodes_dp[2 * N_QUAD];
  __shared__ T weights[2 * N_QUAD];
  __shared__ T kernel_values[N_QUAD];
  __shared__ T nodes[N_QUAD];

  legendre_compute_glr(2 * N_QUAD, nodes_dp, weights);

  if (threadIdx.x < N_QUAD) {
    nodes[threadIdx.x] = T(nodes_dp[threadIdx.x]) * half_width;  // upscale nodes to [0, half_width]
    // the factor 2 is pulled out from the loop over all locations, which
    // calculates 2*cos(..) *...
    kernel_values[threadIdx.x] =
        2 * half_width * T(weights[threadIdx.x]) * kernel.eval_scalar(nodes[threadIdx.x]);
  }
  __syncthreads();

  const auto num_out = loc[0].size();

  for (IntType idx_out = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx_out < num_out;
       idx_out += BLOCK_SIZE * gridDim.x) {
    T phi_hat = 1;
	T phase = 0;
    for (IntType dim = 0; dim < DIM; ++dim) {
      // NOTE modified to apply translation + scaling for type 3 output
      // locations

      const T loc_translated = (loc[dim][idx_out] - out_offsets[dim]);
      const T loc_rescaled = loc_translated * scaling_factors[dim];
      T sum = 0;
      for (IntType idx_quad = 0; idx_quad < N_QUAD; ++idx_quad) {
        sum += kernel_values[idx_quad] * calc_cos(loc_rescaled * nodes[idx_quad]);
      }
      phi_hat *= sum;
      phase += (loc_translated)*in_offsets[dim];
    }
    postphase[idx_out] =
        ComplexType<T>{calc_cos(phase) / phi_hat, sign * calc_sin(phase) / phi_hat};
  }
}

template <typename T, IntType DIM, int N_SPREAD>
static auto postphase_dispatch(const api::DevicePropType& prop, const api::StreamType& stream,
                        const KernelParameters<T>& param, T sign,
                        StackArray<ConstDeviceView<T, 1>, DIM> loc, StackArray<T, DIM> in_offsets,
                        StackArray<T, DIM> out_offsets, StackArray<T, DIM> scaling_factors,
                        DeviceView<ComplexType<T>, 1> postphase) -> void {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  constexpr int BLOCK_SIZE = 256;

  if (param.n_spread == N_SPREAD) {
    EsKernelDirect<T, N_SPREAD> kernel{param.es_beta};
    const dim3 block_dim(BLOCK_SIZE, 1, 1);
    const auto grid_dim = kernel_launch_grid(prop, {postphase.size(), 1, 1}, block_dim);

    api::launch_kernel(postphase_kernel<T, DIM, decltype(kernel), BLOCK_SIZE>, grid_dim, block_dim,
                       0, stream, kernel, sign, loc, in_offsets, out_offsets, scaling_factors,
                       postphase);

  } else {
    if constexpr (N_SPREAD > 2) {
      postphase_dispatch<T, DIM, N_SPREAD - 1>(prop, stream, param, sign, loc, in_offsets,
                                               out_offsets, scaling_factors, postphase);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}


template <typename T, IntType DIM>
auto postphase(const api::DevicePropType& prop, const api::StreamType& stream,
                        const KernelParameters<T>& param, T sign,
                        StackArray<ConstDeviceView<T, 1>, DIM> loc, StackArray<T, DIM> in_offsets,
                        StackArray<T, DIM> out_offsets, StackArray<T, DIM> scaling_factors,
                        DeviceView<ComplexType<T>, 1> postphase) -> void {
  postphase_dispatch<T, DIM, 16>(prop, stream, param, sign, loc, in_offsets, out_offsets,
                                 scaling_factors, postphase);
}

template auto postphase<float, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  const KernelParameters<float>& param, float sign,
                                  StackArray<ConstDeviceView<float, 1>, 1> loc,
                                  StackArray<float, 1> in_offsets, StackArray<float, 1> out_offsets,
                                  StackArray<float, 1> scaling_factors,
                                  DeviceView<ComplexType<float>, 1> postphase) -> void;

template auto postphase<float, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  const KernelParameters<float>& param, float sign,
                                  StackArray<ConstDeviceView<float, 1>, 2> loc,
                                  StackArray<float, 2> in_offsets, StackArray<float, 2> out_offsets,
                                  StackArray<float, 2> scaling_factors,
                                  DeviceView<ComplexType<float>, 1> postphase) -> void;

template auto postphase<float, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  const KernelParameters<float>& param, float sign,
                                  StackArray<ConstDeviceView<float, 1>, 3> loc,
                                  StackArray<float, 3> in_offsets, StackArray<float, 3> out_offsets,
                                  StackArray<float, 3> scaling_factors,
                                  DeviceView<ComplexType<float>, 1> postphase) -> void;

template auto postphase<double, 1>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  const KernelParameters<double>& param, double sign,
                                  StackArray<ConstDeviceView<double, 1>, 1> loc,
                                  StackArray<double, 1> in_offsets, StackArray<double, 1> out_offsets,
                                  StackArray<double, 1> scaling_factors,
                                  DeviceView<ComplexType<double>, 1> postphase) -> void;

template auto postphase<double, 2>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  const KernelParameters<double>& param, double sign,
                                  StackArray<ConstDeviceView<double, 1>, 2> loc,
                                  StackArray<double, 2> in_offsets, StackArray<double, 2> out_offsets,
                                  StackArray<double, 2> scaling_factors,
                                  DeviceView<ComplexType<double>, 1> postphase) -> void;

template auto postphase<double, 3>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  const KernelParameters<double>& param, double sign,
                                  StackArray<ConstDeviceView<double, 1>, 3> loc,
                                  StackArray<double, 3> in_offsets, StackArray<double, 3> out_offsets,
                                  StackArray<double, 3> scaling_factors,
                                  DeviceView<ComplexType<double>, 1> postphase) -> void;

}  // namespace gpu
}  // namespace neonufft
