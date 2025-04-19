#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "neonufft/config.h"

#include "contrib/legendre_rule/legendre_rule_fast.hpp"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "neonufft/kernels/fseries_inverse_kernel.cpp" // this file

#include "neonufft/kernels/hwy_dispatch.hpp"

// must be included after highway headers
#include "neonufft/kernels/es_kernel_eval.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE {  // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename KER, typename T>
HWY_ATTR void fseries_inverse_kernel(const KER& kernel, IntType grid_size, T* HWY_RESTRICT fs) {
  constexpr auto N_SPREAD = KER::N_SPREAD;
  constexpr auto N_QUAD = N_SPREAD + (N_SPREAD % 2) + 10;
  constexpr T half_width = T(N_SPREAD) / T(2);

  const TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);

  std::array<double, 2 * N_QUAD> nodes_dp;
  std::array<double, 2 * N_QUAD> weights;
  HWY_ALIGN std::array<T, N_QUAD + (N_QUAD % n_lanes)> kernel_values = {0};
  HWY_ALIGN std::array<T, N_QUAD + (N_QUAD % n_lanes)> phases = {0};

  // kernel is symmetric. Use only nodes in [0, 1]
  // Always in double precision
  contrib::quadrature::legendre_compute_glr(2 * N_QUAD, nodes_dp.data(), weights.data());

  constexpr T width = N_SPREAD;
  const T alpha = math::pi<T> * width / grid_size;
  for (IntType idx_quad = 0; idx_quad < N_QUAD; ++idx_quad) {
    const T node_scaled = T(nodes_dp[idx_quad]) * half_width;  // upscale nodes to [0, half_width]
    kernel_values[idx_quad] = width * T(weights[idx_quad]) * kernel.eval_scalar(node_scaled);
    phases[idx_quad] = alpha * T(nodes_dp[idx_quad]);
  }


  const IntType n_out = grid_size / 2 + 1;

  for (IntType idx_out = 0; idx_out < n_out; ++idx_out) {
    const T sign = (idx_out % 2) ? T(-1) : T(1);
    const auto v_idx_out_flt = hn::Set(d, idx_out);

    auto v_sum = hn::Mul(hn::Load(d, kernel_values.data()),
                         hn::Cos(d, hn::Mul(v_idx_out_flt, hn::Load(d, phases.data()))));
    for (IntType idx_quad = n_lanes; idx_quad < N_QUAD; idx_quad += n_lanes) {
      v_sum = hn::Add(
          v_sum,
          hn::Mul(hn::Load(d, kernel_values.data() + idx_quad),
                  hn::Cos(d, hn::Mul(v_idx_out_flt, hn::Load(d, phases.data() + idx_quad)))));
    }
    auto sum = hn::ReduceSum(d, v_sum);

    fs[idx_out] = sign / sum;
  }

  // Original. Accuracy issues if grid_size is power of 2
  /*
  std::vector<T> f(N_QUAD);
  std::vector<double> z(2 * N_QUAD);
  std::vector<double> w(2 * N_QUAD);
  std::vector<std::complex<T>> a(N_QUAD);
  contrib::quadrature::legendre_compute_glr(
      2 * N_QUAD, z.data(),
      w.data()); // only half the nodes used, eg on (0,1)

  const IntType nfHalf = grid_size / 2;
  for (int n = 0; n < N_QUAD; ++n) { // set up nodes z_n and vals f_n
    z[n] *= half_width;                 // rescale nodes
    f[n] = half_width * (T)w[n] * kernel.eval_scalar((T)z[n]);  // vals & quadr wei
    a[n] = std::exp(T(2 * math::pi<T>) * std::complex<T>{0, 1} * (T)(nfHalf - z[n]) /
               (T)grid_size); // phase winding rates
  }

  std::vector<std::complex<T>> aj(N_QUAD); // phase rotator for this thread
  for (int n = 0; n < N_QUAD; ++n)
    aj[n] = 1;

  for (IntType j = 0; j < n_out; ++j) { // loop along output array
    T x = 0.0;                         // accumulator for answer at this j
    for (int n = 0; n < N_QUAD; ++n) {
      x += f[n] * 2 * real(aj[n]); // include the negative freq
      aj[n] *= a[n];               // wind the phases
    }
    fs[j] = T(1) / x; // NOTE: modified to compute inverse
  }
  */

}

template <typename T, IntType N_SPREAD>
HWY_ATTR void fseries_inverse_dispatch(NeonufftKernelType kernel_type,
                                       const KernelParameters<T>& kernel_param, IntType grid_size,
                                       T* HWY_RESTRICT fs) {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  if (kernel_param.n_spread == N_SPREAD) {
    if (kernel_type == NEONUFFT_ES_KERNEL) {
      if (kernel_param.approximation && kernel_param.upsampfac == 2.0) {
        EsKernelHorner200<T, N_SPREAD> kernel(kernel_param);
        fseries_inverse_kernel<decltype(kernel), T>(kernel, grid_size, fs);

      } else if (kernel_param.approximation && kernel_param.upsampfac == 1.25) {
        EsKernelHorner125<T, N_SPREAD> kernel(kernel_param);
        fseries_inverse_kernel<decltype(kernel), T>(kernel, grid_size, fs);
      } else {
        EsKernelDirect<T, N_SPREAD> kernel(kernel_param);
        fseries_inverse_kernel<decltype(kernel), T>(kernel, grid_size, fs);
      }
    } else {
      throw InternalError("Unknown kernel type");
    }
  } else {
    if constexpr (N_SPREAD > 2) {
      fseries_inverse_dispatch<T, N_SPREAD - 1>(kernel_type, kernel_param, grid_size, fs);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}

template <typename T>
HWY_ATTR void fseries_inverse_dispatch_init(NeonufftKernelType kernel_type,
                                            const KernelParameters<T>& kernel_param,
                                            IntType grid_size, T* HWY_RESTRICT fs) {
  fseries_inverse_dispatch<T, 16>(kernel_type, kernel_param, grid_size, fs);
}

}  // namespace HWY_NAMESPACE
}  // namespace

#if HWY_ONCE

template <typename T>
void fseries_inverse(NeonufftKernelType kernel_type, const KernelParameters<T>& kernel_param,
                     IntType grid_size, T*  fs) {
  NEONUFFT_EXPORT_AND_DISPATCH_T(fseries_inverse_dispatch_init<T>)
  (kernel_type, kernel_param, grid_size, fs);
}

template void fseries_inverse<float>(NeonufftKernelType kernel_type,
                                        const KernelParameters<float>& kernel_param,
                                        IntType grid_size, float*  fs

);

template void fseries_inverse<double>(NeonufftKernelType kernel_type,
                                         const KernelParameters<double>& kernel_param,
                                         IntType grid_size, double*  fs

);

#endif

}  // namespace neonufft
