#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "neonufft/config.h"

#include "neonufft/enums.h"
#include "neonufft/types.hpp"
#include "util/math.hpp"
// #include "kernels/upsample_kernel.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/compute_prephase_kernel.cpp" // this file
                                                                 //
#include "kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template<typename T>
constexpr bool IsAligned(T* ptr, size_t align = 128) {
  return reinterpret_cast<uintptr_t>(ptr) % align == 0;
}

template <typename T, IntType DIM>
HWY_ATTR void
compute_prephase_kernel(int sign, IntType n, std::array<const T*, DIM> in_loc, std::array<T, DIM> out_offset,
                           std::complex<T> *HWY_RESTRICT prephase_aligned) {

  T*HWY_RESTRICT prephase_aligned_scalar = reinterpret_cast<T*>(prephase_aligned);

  const TagType<T> d;
  const IntType n_lanes = hn::Lanes(d);

  const auto v_sign = hn::Set(d, T(sign));
  const auto v_out_offset_x = hn::Set(d, out_offset[0]);
  auto v_out_offset_y = hn::Undefined(d);
  auto v_out_offset_z = hn::Undefined(d);

  if constexpr (DIM >= 2) {
    v_out_offset_y = hn::Set(d, out_offset[1]);
  }
  if constexpr (DIM >= 3) {
    v_out_offset_z = hn::Set(d, out_offset[2]);
  }

  IntType idx = 0;
  for(; idx + n_lanes <= n; idx += n_lanes) {
    auto phase = hn::Mul(hn::LoadU(d, in_loc[0] + idx), v_out_offset_x);
    if constexpr(DIM >= 2) {
      phase = hn::Add(phase, hn::Mul(hn::LoadU(d, in_loc[1] + idx), v_out_offset_y));
    }
    if constexpr(DIM >= 3) {
      phase = hn::Add(phase, hn::Mul(hn::LoadU(d, in_loc[2] + idx), v_out_offset_z));
    }

    const auto real = hn::Cos(d, phase);
    const auto imag = hn::Mul(v_sign, hn::Sin(d, phase));

    const auto cpx_1 = hn::InterleaveWholeLower(d, real, imag);
    const auto cpx_2 = hn::InterleaveWholeUpper(d, real, imag);

    hn::Store(cpx_1, d, prephase_aligned_scalar + idx * 2);
    hn::Store(cpx_2, d, prephase_aligned_scalar + idx * 2 + n_lanes);
  }

  for (; idx < n; ++idx) {
    auto phase = in_loc[0][idx] * out_offset[0];
    if constexpr (DIM >= 2) {
      phase += in_loc[1][idx] * out_offset[1];
    }
    if constexpr (DIM >= 3) {
      phase += in_loc[2][idx] * out_offset[2];
    }
    prephase_aligned[idx] = std::complex<T>{std::cos(phase), sign * std::sin(phase)};
  }
}

template <IntType DIM>
HWY_ATTR void compute_prephase_kernel_float(
    int sign, IntType n, std::array<const float *, DIM> in_loc,
    std::array<float, DIM> out_offset,
    std::complex<float> *HWY_RESTRICT prephase_aligned) {
  compute_prephase_kernel<float, DIM>(sign, n, in_loc, out_offset, prephase_aligned);
}

template <IntType DIM>
HWY_ATTR void compute_prephase_kernel_double(
    int sign, IntType n, std::array<const double *, DIM> in_loc,
    std::array<double, DIM> out_offset,
    std::complex<double> *HWY_RESTRICT prephase_aligned) {
  compute_prephase_kernel<double, DIM>(sign, n, in_loc, out_offset, prephase_aligned);
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void compute_prephase(int sign, IntType n, std::array<const T *, DIM> in_loc,
                      std::array<T, DIM> out_offset,
                      std::complex<T> *prephase_aligned) {
  if constexpr (std::is_same_v<T, float>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(compute_prephase_kernel_float<DIM>)
    (sign, n, in_loc, out_offset, prephase_aligned);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(compute_prephase_kernel_double<DIM>)
    (sign, n, in_loc, out_offset, prephase_aligned);
  }
}

template void compute_prephase<float, 1>(int sign, IntType n,
                                         std::array<const float *, 1> in_loc,
                                         std::array<float, 1> out_offset,
                                         std::complex<float> *prephase_aligned);

template void compute_prephase<float, 2>(int sign, IntType n,
                                         std::array<const float *, 2> in_loc,
                                         std::array<float, 2> out_offset,
                                         std::complex<float> *prephase_aligned);

template void compute_prephase<float, 3>(int sign, IntType n,
                                         std::array<const float *, 3> in_loc,
                                         std::array<float, 3> out_offset,
                                         std::complex<float> *prephase_aligned);

template void compute_prephase<double, 1>(
    int sign, IntType n, std::array<const double *, 1> in_loc,
    std::array<double, 1> out_offset, std::complex<double> *prephase_aligned);

template void compute_prephase<double, 2>(
    int sign, IntType n, std::array<const double *, 2> in_loc,
    std::array<double, 2> out_offset, std::complex<double> *prephase_aligned);

template void compute_prephase<double, 3>(
    int sign, IntType n, std::array<const double *, 3> in_loc,
    std::array<double, 3> out_offset, std::complex<double> *prephase_aligned);
#endif

} // namespace neonufft
