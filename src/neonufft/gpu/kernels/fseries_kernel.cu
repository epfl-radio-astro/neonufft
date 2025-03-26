#include "neonufft/config.h"
//---
#include <algorithm>
#include <cassert>

#include "neonufft/enums.h"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/gpu/kernels/es_kernel_eval.cuh"
#include "neonufft/gpu/kernels/legendre_rule.cuh"
#include "neonufft/gpu/kernels/fseries_kernel.hpp"
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

template <typename T, typename KER, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    fseries_inverse_kernel(KER kernel, IntType grid_size,
                           DeviceView<T, 1> fseries_inverse) {
  constexpr auto N_SPREAD = KER::n_spread;
  constexpr auto N_QUAD = N_SPREAD + (N_SPREAD % 2);
  // constexpr int N_QUAD = 2 + 3.0 * (double(N_SPREAD) / 2);
  constexpr T half_width = T(N_SPREAD) / T(2);
  constexpr T two_pi = math::pi<T> / 2;

  static_assert(BLOCK_SIZE >= N_QUAD);

  const IntType n_out = grid_size / 2 + 1;
  assert(n_out == fseries_inverse.size());

  __shared__ double nodes_dp[2 * N_QUAD];
  __shared__ double weights[2 * N_QUAD];
  __shared__ T phases[N_QUAD];
  __shared__ T kernel_values[N_QUAD];

  legendre_compute_glr(2 * N_QUAD, nodes_dp, weights);

  constexpr T width = N_SPREAD;
  const T alpha = math::pi<T> * width / grid_size;
  if (threadIdx.x < N_QUAD) {
    const T node_scaled = T(nodes_dp[threadIdx.x]) * half_width;  // upscale nodes to [0, half_width]
    kernel_values[threadIdx.x] = width * T(weights[threadIdx.x]) * kernel.eval_scalar(node_scaled);
    phases[threadIdx.x] = alpha * T(nodes_dp[threadIdx.x]);
  }
  __syncthreads();

  for (IntType idx_out = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx_out < n_out;
       idx_out += BLOCK_SIZE * gridDim.x) {
    const T sign = (idx_out % 2) ? T(-1) : T(1);
    const T idx_out_flt = idx_out;

    T sum = 0;
    for (IntType idx_quad = 0; idx_quad < N_QUAD; ++idx_quad) {
      sum += kernel_values[idx_quad] * calc_cos(idx_out_flt * phases[idx_quad]);
    }

    fseries_inverse[idx_out] = sign / sum;
  }
}

template <typename T, int N_SPREAD>
static auto fseries_inverse_dispatch(const api::DevicePropType& prop, const api::StreamType& stream,
                                     const KernelParameters<T>& param, IntType grid_size,
                                     DeviceView<T, 1> fseries_inverse) -> void {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  constexpr int BLOCK_SIZE = 256;

  if (param.n_spread == N_SPREAD) {
    const IntType n_out = grid_size / 2 + 1;
    assert(n_out == fseries_inverse.size());
    EsKernelDirect<T, N_SPREAD> kernel{param.es_beta};
    const dim3 block_dim(BLOCK_SIZE, 1, 1);
    const auto grid_dim = kernel_launch_grid(prop, {n_out, 1, 1}, block_dim);

    api::launch_kernel(fseries_inverse_kernel<T, decltype(kernel), BLOCK_SIZE>, grid_dim,
                       block_dim, 0, stream, kernel, grid_size, fseries_inverse);

  } else {
    if constexpr (N_SPREAD > 2) {
      fseries_inverse_dispatch<T, N_SPREAD - 1>(prop, stream, param, grid_size,
                                                     fseries_inverse);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}

template <typename T>
auto fseries_inverse(const api::DevicePropType& prop, const api::StreamType& stream,
                     const KernelParameters<T>& param, IntType grid_size,
                     DeviceView<T, 1> fseries_inverse) -> void {
  fseries_inverse_dispatch<T, 16>(prop, stream, param, grid_size, fseries_inverse);
}

template auto fseries_inverse<float>(const api::DevicePropType& prop, const api::StreamType& stream,
                                     const KernelParameters<float>& param, IntType grid_size,
                                     DeviceView<float, 1> fseries_inverse) -> void;

template auto fseries_inverse<double>(const api::DevicePropType& prop,
                                      const api::StreamType& stream,
                                      const KernelParameters<double>& param, IntType grid_size,
                                      DeviceView<double, 1> fseries_inverse) -> void;

}  // namespace gpu
}  // namespace neonufft
