#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "neonufft/config.h"

#include "legendre_rule/legendre_rule_fast.hpp"
#include "es_kernel_param.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/nuft_real_kernel.cpp" // this file

#include "kernels/hwy_dispatch.hpp"

// must be included after highway headers
#include "kernels/es_kernel_eval.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <IntType DIM, typename KER, typename T>
HWY_ATTR void nuft_real_kernel(const KER &kernel, IntType num_in,
                               std::array<const T *, DIM> loc,
                               std::array<T, DIM> offsets,
                               std::array<T, DIM> scaling_factors,
                               T *HWY_RESTRICT phi_had_aligned) {

  constexpr auto N_SPREAD = KER::N_SPREAD;
  constexpr auto N_QUAD = N_SPREAD + (N_SPREAD % 2);
  constexpr T half_width = T(N_SPREAD) / T(2);

  const TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);

  std::array<double, 2 * N_QUAD> nodes_dp;
  std::array<double, 2 * N_QUAD> weights;
  std::array<T, N_QUAD> kernel_values;

  // kernel is symmetric. Use only nodes in [0, 1]
  // Always in double precision
  contrib::quadrature::legendre_compute_glr(2 * N_QUAD, nodes_dp.data(),
                                            weights.data());

  // Copy to type T
  std::array<T, 2 * N_QUAD> nodes;
  std::copy(nodes_dp.begin(), nodes_dp.end(), nodes.begin());

  for (IntType idx_quad = 0; idx_quad < N_QUAD; ++idx_quad) {
    nodes[idx_quad] *= (T)half_width; // upscale nodes to [0, half_width]
    // the factor 2 is pulled out from the loop over all locations, which
    // calculates 2*cos(..) *...
    kernel_values[idx_quad] = 2 * half_width * T(weights[idx_quad]) *
                              kernel.eval_scalar(nodes[idx_quad]);
  }

  IntType idx_in = 0;
  for (; idx_in + n_lanes <= num_in; idx_in += n_lanes) {
    auto v_prod = hn::Set(d, 1);

    for (IntType dim = 0; dim < DIM; ++dim) {
      // NOTE modified to apply translation + scaling for type 3 output
      // locations
      const auto v_loc_rescaled = hn::Mul(
          hn::Sub(hn::LoadU(d, loc[dim] + idx_in), hn::Set(d, offsets[dim])),
          hn::Set(d, scaling_factors[dim]));
      auto v_sum = hn::Set(d, 0);
      for (IntType idx_quad = 0; idx_quad < N_QUAD; ++idx_quad) {
        v_sum = hn::MulAdd(
            hn::Set(d, kernel_values[idx_quad]),
            hn::Cos(d, hn::Mul(v_loc_rescaled, hn::Set(d, nodes[idx_quad]))),
            v_sum);
      }
      v_prod = hn::Mul(v_prod, v_sum);
    }

    hn::Store(v_prod, d, phi_had_aligned + idx_in);
  }

  for (; idx_in < num_in; ++idx_in) {
    T prod = 1;
    for (IntType dim = 0; dim < DIM; ++dim) {
      // NOTE modified to apply translation + scaling for type 3 output
      // locations
      const T rescaled_coord =
          (loc[dim][idx_in] - offsets[dim]) * scaling_factors[dim];
      T sum = 0;
      for (IntType idx_quad = 0; idx_quad < N_QUAD; ++idx_quad) {
        sum += kernel_values[idx_quad] *
             std::cos(rescaled_coord * nodes[idx_quad]);
      }
      prod *= sum;
    }
    phi_had_aligned[idx_in] = prod;
  }
}

template <IntType DIM, typename T, IntType N_SPREAD>
HWY_ATTR void
nuft_real_dispatch(NeonufftKernelType kernel_type,
                   const KernelParameters<T> &kernel_param, IntType num_in,
                   std::array<const T *, DIM> loc, std::array<T, DIM> offsets,
                   std::array<T, DIM> scaling_factors, T *phi_had_aligned) {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  if (kernel_param.n_spread == N_SPREAD) {
    if (kernel_type == NEONUFFT_ES_KERNEL) {
      if (kernel_param.approximation && kernel_param.upsampfac == 2.0) {
        EsKernelHorner200<T, N_SPREAD> kernel(kernel_param);
        nuft_real_kernel<DIM, decltype(kernel), T>(
            kernel, num_in, loc, offsets, scaling_factors, phi_had_aligned);

      } else if (kernel_param.approximation && kernel_param.upsampfac == 1.25) {
        EsKernelHorner125<T, N_SPREAD> kernel(kernel_param);
        nuft_real_kernel<DIM, decltype(kernel), T>(
            kernel, num_in, loc, offsets, scaling_factors, phi_had_aligned);
      } else {
        EsKernelDirect<T, N_SPREAD> kernel(kernel_param);
        nuft_real_kernel<DIM, decltype(kernel), T>(
            kernel, num_in, loc, offsets, scaling_factors, phi_had_aligned);
      }
    } else {
      throw InternalError("Unknown kernel type");
    }
  } else {
    if constexpr (N_SPREAD > 2) {
      nuft_real_dispatch<DIM, T, N_SPREAD - 1>(
          kernel_type, kernel_param, num_in, loc, offsets, scaling_factors,
          phi_had_aligned);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}

template <IntType DIM>
HWY_ATTR void nuft_real_float(NeonufftKernelType kernel_type,
                              const KernelParameters<float> &kernel_param,
                              IntType num_in,
                              std::array<const float *, DIM> loc,
                              std::array<float, DIM> offsets,
                              std::array<float, DIM> scaling_factors,
                              float *phi_had_aligned) {
  nuft_real_dispatch<DIM, float, 16>(kernel_type, kernel_param, num_in, loc,
                                     offsets, scaling_factors, phi_had_aligned);
}

template <IntType DIM>
HWY_ATTR void nuft_real_double(NeonufftKernelType kernel_type,
                               const KernelParameters<double> &kernel_param,
                               IntType num_in,
                               std::array<const double *, DIM> loc,
                               std::array<double, DIM> offsets,
                               std::array<double, DIM> scaling_factors,
                               double *phi_had_aligned) {
  nuft_real_dispatch<DIM, double, 16>(kernel_type, kernel_param, num_in, loc,
                                      offsets, scaling_factors,
                                      phi_had_aligned);
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void nuft_real(NeonufftKernelType kernel_type,
               const KernelParameters<T> &kernel_param, IntType num_in,
               std::array<const T *, DIM> loc, std::array<T, DIM> offsets,
               std::array<T, DIM> scaling_factors, T *phi_had_aligned) {
  if constexpr (std::is_same_v<T, float>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(nuft_real_float<DIM>)
    (kernel_type, kernel_param, num_in, loc, offsets, scaling_factors,
     phi_had_aligned);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(nuft_real_double<DIM>)
    (kernel_type, kernel_param, num_in, loc, offsets, scaling_factors,
     phi_had_aligned);
  }
}

template void nuft_real<float, 1>(NeonufftKernelType kernel_type,
                                  const KernelParameters<float> &kernel_param,
                                  IntType num_in,
                                  std::array<const float *, 1> loc,
                                  std::array<float, 1> offsets,
                                  std::array<float, 1> scaling_factors,
                                  float *phi_had_aligned);

template void nuft_real<float, 2>(NeonufftKernelType kernel_type,
                                  const KernelParameters<float> &kernel_param,
                                  IntType num_in,
                                  std::array<const float *, 2> loc,
                                  std::array<float, 2> offsets,
                                  std::array<float, 2> scaling_factors,
                                  float *phi_had_aligned);

template void nuft_real<float, 3>(NeonufftKernelType kernel_type,
                                  const KernelParameters<float> &kernel_param,
                                  IntType num_in,
                                  std::array<const float *, 3> loc,
                                  std::array<float, 3> offsets,
                                  std::array<float, 3> scaling_factors,
                                  float *phi_had_aligned);

template void nuft_real<double, 1>(NeonufftKernelType kernel_type,
                                   const KernelParameters<double> &kernel_param,
                                   IntType num_in,
                                   std::array<const double *, 1> loc,
                                   std::array<double, 1> offsets,
                                   std::array<double, 1> scaling_factors,
                                   double *phi_had_aligned);

template void nuft_real<double, 2>(NeonufftKernelType kernel_type,
                                   const KernelParameters<double> &kernel_param,
                                   IntType num_in,
                                   std::array<const double *, 2> loc,
                                   std::array<double, 2> offsets,
                                   std::array<double, 2> scaling_factors,
                                   double *phi_had_aligned);

template void nuft_real<double, 3>(NeonufftKernelType kernel_type,
                                   const KernelParameters<double> &kernel_param,
                                   IntType num_in,
                                   std::array<const double *, 3> loc,
                                   std::array<double, 3> offsets,
                                   std::array<double, 3> scaling_factors,
                                   double *phi_had_aligned);

#endif

} // namespace neonufft
