#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <complex>

#include "neonufft/config.h"

#include "es_kernel_param.hpp"
#include "kernels/compute_postphase_kernel.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"
#include "util/math.hpp"
// #include "kernels/upsample_kernel.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/compute_postphase_kernel.cpp" // this file

#include "kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T, IntType DIM>
HWY_ATTR void compute_postphase_kernel(
    int sign, IntType n, const T *HWY_RESTRICT phi_hat_aligned,
    std::array<const T *, DIM> out_loc, std::array<T, DIM> in_offset,
    std::array<T, DIM> out_offset,
    std::complex<T> *HWY_RESTRICT postphase_aligned) {

  T*HWY_RESTRICT postphase_aligned_scalar = reinterpret_cast<T*>(postphase_aligned);

  const TagType<T> d;
  const IntType n_lanes = hn::Lanes(d);

  const auto v_sign = hn::Set(d, T(sign));
  const auto v_in_offset_x = hn::Set(d, in_offset[0]);
  const auto v_out_offset_x = hn::Set(d, out_offset[0]);

  auto v_in_offset_y = hn::Undefined(d);
  auto v_out_offset_y = hn::Undefined(d);
  auto v_in_offset_z = hn::Undefined(d);
  auto v_out_offset_z = hn::Undefined(d);


  if constexpr (DIM >= 2) {
    v_in_offset_y = hn::Set(d, in_offset[1]);
    v_out_offset_y = hn::Set(d, out_offset[1]);
  }
  if constexpr (DIM >= 3) {
    v_in_offset_z = hn::Set(d, in_offset[2]);
    v_out_offset_z = hn::Set(d, out_offset[2]);
  }


  IntType idx = 0;
  for(; idx + n_lanes <= n; idx += n_lanes) {
    auto phase = hn::Mul(
        hn::Sub(hn::LoadU(d, out_loc[0] + idx), v_out_offset_x), v_in_offset_x);
    if constexpr(DIM >= 2) {
      phase = hn::Add(phase, hn::Mul(hn::Sub(hn::LoadU(d, out_loc[1] + idx),
                                             v_out_offset_y),
                                     v_in_offset_y));
    }
    if constexpr(DIM >= 3) {
      phase = hn::Add(phase, hn::Mul(hn::Sub(hn::LoadU(d, out_loc[2] + idx),
                                             v_out_offset_z),
                                     v_in_offset_z));
    }

    auto real = hn::Cos(d, phase);
    auto imag = hn::Mul(v_sign, hn::Sin(d, phase));

    const auto v_phi_hat = hn::Load(d, phi_hat_aligned + idx);
    real = hn::Div(real, v_phi_hat);
    imag = hn::Div(imag, v_phi_hat);

    hn::Store(hn::InterleaveWholeLower(d, real, imag), d,
              postphase_aligned_scalar + idx * 2);
    hn::Store(hn::InterleaveWholeUpper(d, real, imag), d,
              postphase_aligned_scalar + idx * 2 + n_lanes);
  }

  for (; idx < n; ++idx) {
    auto phase = (out_loc[0][idx] - out_offset[0]) * in_offset[0];
    if constexpr(DIM >= 2) {
      phase += (out_loc[1][idx] - out_offset[1]) * in_offset[1];
    }
    if constexpr(DIM >= 3) {
      phase += (out_loc[2][idx] - out_offset[2]) * in_offset[2];
    }
    const auto pp = std::complex<T>{std::cos(phase), sign * std::sin(phase)} /
                    phi_hat_aligned[idx];
    postphase_aligned[idx] = pp;
  }
}

template <IntType DIM>
HWY_ATTR void compute_postphase_kernel_float(
    int sign, IntType n, const float *phi_hat_aligned,
    std::array<const float *, DIM> out_loc, std::array<float, DIM> in_offset,
    std::array<float, DIM> out_offset, std::complex<float> *postphase_aligned) {
  compute_postphase_kernel<float, DIM>(sign, n, phi_hat_aligned, out_loc,
                                       in_offset, out_offset,
                                       postphase_aligned);
}

template <IntType DIM>
HWY_ATTR void compute_postphase_kernel_double(
    int sign, IntType n, const double *phi_hat_aligned,
    std::array<const double *, DIM> out_loc, std::array<double, DIM> in_offset,
    std::array<double, DIM> out_offset, std::complex<double> *postphase_aligned) {
  compute_postphase_kernel<double, DIM>(sign, n, phi_hat_aligned, out_loc,
                                        in_offset, out_offset,
                                        postphase_aligned);
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void compute_postphase(int sign, IntType n, const T *phi_hat_aligned,
                       std::array<const T *, DIM> out_loc,
                       std::array<T, DIM> in_offset,
                       std::array<T, DIM> out_offset,
                       std::complex<T> *postphase_aligned) {
  if constexpr (std::is_same_v<T, float>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(compute_postphase_kernel_float<DIM>)
    (sign, n, phi_hat_aligned, out_loc, in_offset, out_offset,
     postphase_aligned);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(compute_postphase_kernel_double<DIM>)
    (sign, n, phi_hat_aligned, out_loc, in_offset, out_offset,
     postphase_aligned);
  }
}

template void compute_postphase<float, 1>(
    int sign, IntType n, const float *phi_hat_aligned,
    std::array<const float *, 1> out_loc, std::array<float, 1> in_offset,
    std::array<float, 1> out_offset, std::complex<float> *postphase_aligned);

template void compute_postphase<float, 2>(
    int sign, IntType n, const float *phi_hat_aligned,
    std::array<const float *, 2> out_loc, std::array<float, 2> in_offset,
    std::array<float, 2> out_offset, std::complex<float> *postphase_aligned);

template void compute_postphase<float, 3>(
    int sign, IntType n, const float *phi_hat_aligned,
    std::array<const float *, 3> out_loc, std::array<float, 3> in_offset,
    std::array<float, 3> out_offset, std::complex<float> *postphase_aligned);

template void compute_postphase<double, 1>(
    int sign, IntType n, const double *phi_hat_aligned,
    std::array<const double *, 1> out_loc, std::array<double, 1> in_offset,
    std::array<double, 1> out_offset, std::complex<double> *postphase_aligned);

template void compute_postphase<double, 2>(
    int sign, IntType n, const double *phi_hat_aligned,
    std::array<const double *, 2> out_loc, std::array<double, 2> in_offset,
    std::array<double, 2> out_offset, std::complex<double> *postphase_aligned);

template void compute_postphase<double, 3>(
    int sign, IntType n, const double *phi_hat_aligned,
    std::array<const double *, 3> out_loc, std::array<double, 3> in_offset,
    std::array<double, 3> out_offset, std::complex<double> *postphase_aligned);

#endif

} // namespace neonufft
