// NOTE: no include guard because of multiple inclusions through highway foreach_target
#include <cassert>
#include <complex>
#include <cstring>
#include <cmath>

#include "neonufft/config.h"

#include "contrib/es_kernel/es_kernel_horner_coeff.hpp"
#include "contrib/es_kernel/es_kernel_horner_coeff_125.hpp"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/memory/array.hpp"
#include "neonufft/memory/copy.hpp"
#include "neonufft/enums.h"
#include "neonufft/exceptions.hpp"
#include "neonufft/types.hpp"

#include "neonufft/kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T, IntType NSPREAD> struct EsKernelDirect {
  inline static constexpr IntType N_SPREAD = NSPREAD;

  HWY_ATTR EsKernelDirect(KernelParameters<T> param) : param(std::move(param)) {
    if (N_SPREAD != param.n_spread) {
      throw InternalError("n_spread template parameter mismatch.");
    }
  }

  HWY_INLINE HWY_ATTR T eval_scalar(T x) const {
    constexpr T es_c = T(4) / T(N_SPREAD * N_SPREAD);
    const T arg = T(1) - es_c * x * x;
    if (arg <= 0) return 0;

    return std::exp(param.es_beta * (std::sqrt(arg) - T(1)));
  }

  // Evaluate kernel at N_SPREAD points starting from x_init in [-w/2, -w/2 + 1]
  // Output buffer must be aligned and padded
  template <typename D>
  HWY_INLINE HWY_ATTR void eval(D d, T x_init, T *HWY_RESTRICT ker) const {
    const IntType n_lanes = hn::Lanes(d);
    const auto inc = hn::Set(d, n_lanes);
    const auto one = hn::Set(d, 1);
    const auto ES_halfwidth = hn::Set(d, param.es_halfwidth);
    const auto ES_beta = hn::Set(d, param.es_beta);
    const auto ES_c = hn::Set(d, param.es_c);

    auto x = hn::Iota(d, x_init); // [x1, x1+1, x1+2, ...]
    for (IntType idx_ker = 0; idx_ker < N_SPREAD; idx_ker += n_lanes) {
      const auto sq_arg =
          hn::ZeroIfNegative(hn::Sub(one, hn::Mul(hn::Mul(x, x), ES_c)));
      const auto exp_arg = hn::Mul(ES_beta, hn::Sub(hn::Sqrt(sq_arg), one));
      hn::Store(hn::Exp(d, exp_arg), d, ker + idx_ker);
      x = hn::Add(x, inc);
    }

    constexpr IntType tail = N_SPREAD % n_lanes;
    if constexpr (tail) {
      for (IntType segment = N_SPREAD; segment < N_SPREAD + (n_lanes - tail); ++segment) {
        ker[segment] = 0;
      }
    }
  }

  // Evaluate kernel at N_SPREAD points starting from x_init in [-w/2, -w/2 + 1]
  // and writing out pairwise duplicate values (x0, x0, x1, x1, ...).
  // Output buffer must be aligned and padded
  template <typename D>
  HWY_INLINE HWY_ATTR void eval2(D d, T x_init, T *HWY_RESTRICT ker) const {
    const IntType n_lanes = hn::Lanes(d);
    const auto inc = hn::Set(d, n_lanes);
    const auto one = hn::Set(d, 1);
    const auto ES_halfwidth = hn::Set(d, param.es_halfwidth);
    const auto ES_beta = hn::Set(d, param.es_beta);
    const auto ES_c = hn::Set(d, param.es_c);

    auto x = hn::Iota(d, x_init); // [x1, x1+1, x1+2, ...]
    for (IntType idx_ker = 0; idx_ker < N_SPREAD; idx_ker += n_lanes) {
      const auto sq_arg =
          hn::ZeroIfNegative(hn::Sub(one, hn::Mul(hn::Mul(x, x), ES_c)));
      const auto exp_arg = hn::Mul(ES_beta, hn::Sub(hn::Sqrt(sq_arg), one));
      const auto res = hn::Exp(d, exp_arg);
      const auto res_low = hn::InterleaveWholeLower(d, res, res);
      const auto res_high = hn::InterleaveWholeUpper(d, res, res);

      hn::Store(res_low, d, ker + 2 * idx_ker);
      hn::Store(res_high, d, ker + 2 * idx_ker + n_lanes);

      x = hn::Add(x, inc);
    }

    constexpr IntType tail = N_SPREAD % n_lanes;
    if constexpr (tail) {
      for (IntType segment = N_SPREAD; segment < N_SPREAD + (n_lanes - tail); ++segment) {
        ker[2 * segment] = 0;
        ker[2 * segment + 1] = 0;
      }
    }
  }

  KernelParameters<T> param;
};

template <typename T, typename COEFFS> struct EsKernelHorner {
  inline static constexpr IntType N_SPREAD = COEFFS::N_SPREAD;

  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  using CoeffArrayType = decltype(COEFFS::values);

  HWY_ATTR EsKernelHorner(KernelParameters<T> param)
      : param(std::move(param)){

    if (N_SPREAD != param.n_spread) {
      throw InternalError("n_spread template parameter mismatch.");
    }
  }

  HWY_INLINE HWY_ATTR T eval_scalar(T x) const {
    // shift from [-halfwidth, halfwidth] to [0, width]
    const T x_ker = x + param.es_halfwidth;

    if (x_ker >= T(N_SPREAD) || x_ker <= 0)
      return 0.0;

    // identify segment
    const IntType segment = std::floor(x_ker);

    // scale to [-1, 1] within segment
    const T x_scaled = (x_ker - T(segment)) * 2 - 1;

    constexpr IntType n_nodes = std::tuple_size<CoeffArrayType>::value;

    T ker = coeff[0][segment];
    for (IntType i = 1; i < n_nodes; ++i) {
      ker = ker * x_scaled + coeff[i][segment];
    }

    return ker;
  }

  // Evaluate kernel at N_SPREAD points starting from x_init in [-w/2, -w/2 + 1]
  // Output buffer must be aligned and padded
  template <typename D>
  HWY_INLINE HWY_ATTR void eval(D d, T x_init, T *HWY_RESTRICT ker) const {
    constexpr IntType n_lanes = hn::Lanes(d);
    constexpr IntType n_nodes = std::tuple_size<CoeffArrayType>::value;

    // each segment interpolation is done in [-1, 1] for better accuracy.
    // x_init is in [-w/2, -w/2+1].
    const auto v_x = hn::Set(d, 2 * x_init + (N_SPREAD - 1));

    for (IntType segment = 0; segment < N_SPREAD; segment += n_lanes) {
      assert(segment + n_lanes <=
             std::tuple_size<typename CoeffArrayType::value_type>::value);

      auto v_ker_1 = hn::Load(d, &(coeff[0][segment]));
      for (IntType i = 1; i < n_nodes; ++i) {
        v_ker_1 = hn::MulAdd(v_ker_1, v_x, hn::Load(d, &(coeff[i][segment])));
      }
      hn::Store(v_ker_1, d, ker + segment);
    }

    constexpr IntType tail = N_SPREAD % n_lanes;
    if constexpr (tail) {
      for (IntType segment = N_SPREAD; segment < N_SPREAD + (n_lanes - tail); ++segment) {
        ker[segment] = 0;
      }
    }
  }

  // Evaluate kernel at N_SPREAD points starting from x_init in [-w/2, -w/2 + 1]
  // and writing out pairwise duplicate values (x0, x0, x1, x1, ...).
  // Output buffer must be aligned and padded
  template <typename D>
  HWY_INLINE HWY_ATTR void eval2(D d, T x_init, T *HWY_RESTRICT ker) const {
    constexpr IntType n_lanes = hn::Lanes(d);
    constexpr IntType n_nodes = std::tuple_size<CoeffArrayType>::value;

    // each segment interpolation is done in [-1, 1] for better accuracy.
    // x_init is in [-w/2, -w/2+1].
    const auto v_x = hn::Set(d, 2 * x_init + (N_SPREAD - 1));

    for (IntType segment = 0; segment < N_SPREAD; segment += n_lanes) {

      assert(segment + n_lanes <=
             std::tuple_size<typename CoeffArrayType::value_type>::value);

      auto v_ker_1 = hn::Load(d, &(coeff[0][segment]));
      for (IntType i = 1; i < n_nodes; ++i) {
        v_ker_1 = hn::MulAdd(v_ker_1, v_x, hn::Load(d, &(coeff[i][segment])));
      }

      auto v_ker_1_low = hn::InterleaveWholeLower(d, v_ker_1, v_ker_1);
      auto v_ker_1_high = hn::InterleaveWholeUpper(d, v_ker_1, v_ker_1);

      hn::Store(v_ker_1_low, d, ker + 2 * segment);
      hn::Store(v_ker_1_high, d, ker + 2 * segment + n_lanes);
    }

    constexpr IntType tail = N_SPREAD % n_lanes;
    if constexpr (tail) {
      for (IntType segment = N_SPREAD; segment < N_SPREAD + (n_lanes - tail); ++segment) {
        ker[2 * segment] = 0;
        ker[2 * segment + 1] = 0;
      }
    }
  }

  inline static const CoeffArrayType &coeff = COEFFS::values;
  KernelParameters<T> param;
};

template <typename T, IntType NSPREAD>
using EsKernelHorner200 = EsKernelHorner<T, contrib::HornerCoeffs<NSPREAD, T>>;

template <typename T, IntType NSPREAD>
using EsKernelHorner125 = EsKernelHorner<T, contrib::HornerCoeffs125<NSPREAD, T>>;

} // namespace HWY_NAMESPACE
} // namespace


} // namespace neonufft
