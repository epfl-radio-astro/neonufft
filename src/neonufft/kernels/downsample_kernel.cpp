#include <array>
#include <cassert>
#include <complex>
#include <cstring>

#include "neonufft/config.h"
#include "neonufft/enums.h"

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/types.hpp"
#include "neonufft/kernels/downsample_kernel.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "neonufft/kernels/downsample_kernel.cpp" // this file

#include "neonufft/kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR void downsample_1d_kernel(NeonufftModeOrder order, T prefac,
                                   IntType n_large,
                                   const std::complex<T> *HWY_RESTRICT in,
                                   IntType n_small, const T *HWY_RESTRICT ker,
                                   std::complex<T> *HWY_RESTRICT out) {
  assert(n_small <= n_large);

  const TagType<T> d;
  const IntType n_lanes = hn::Lanes(d);

  const auto v_prefac = hn::Set(d, prefac);

  const IntType n_negative = n_small / 2; // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;// [0, N/2 -1]

  IntType idx_out_non_negative = n_negative;
  IntType idx_out_negative = 0;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_out_non_negative = 0;
    idx_out_negative = n_non_negative;
  }

  auto in_scalar = reinterpret_cast<const T *>(in);
  auto out_scalar = reinterpret_cast<T *>(out);

  {
    IntType i = 0;
    for (; i + n_lanes <= n_non_negative;
         i += n_lanes, idx_out_non_negative += n_lanes) {
      // duplicate each ker value to multiply with real and imag interleaved
      auto v_ker = hn::LoadU(d, ker + i);
      auto v_ker_1 = hn::InterleaveWholeLower(d, v_ker, v_ker);
      auto v_ker_2 = hn::InterleaveWholeUpper(d, v_ker, v_ker);

      // load complex values as scalars, since we only multiply with real values
      auto in_current = in_scalar + 2 * i;
      auto v_in_1 = hn::LoadU(d, in_current);
      auto v_in_2 = hn::LoadU(d, in_current + n_lanes);

      auto out_current = out_scalar + 2 * idx_out_non_negative;
      hn::StoreU(hn::Mul(hn::Mul(v_in_1, v_prefac), v_ker_1), d, out_current);
      hn::StoreU(hn::Mul(hn::Mul(v_in_2, v_prefac), v_ker_2), d,
                 out_current + n_lanes);
    }

    for (; i < n_non_negative; ++i, ++idx_out_non_negative) {
      out[idx_out_non_negative] = prefac * in[i] * ker[i];
    }
  }

  const IntType offset_negative = n_large - n_negative;

  {
    IntType i = 0;
    for (; i + n_lanes <= n_negative;
         i += n_lanes, idx_out_negative += n_lanes) {
      // duplicate each ker value to multiply with real and imag interleaved
      // reverse because ker is read backwards
      auto v_ker =
          hn::Reverse(d, hn::LoadU(d, ker + n_negative - i - n_lanes + 1));
      auto v_ker_1 = hn::InterleaveWholeLower(d, v_ker, v_ker);
      auto v_ker_2 = hn::InterleaveWholeUpper(d, v_ker, v_ker);

      // load complex values as scalars, since we only multiply with real values
      auto in_current = in_scalar +2 * (i + offset_negative);
      auto v_in_1 = hn::LoadU(d, in_current);
      auto v_in_2 = hn::LoadU(d, in_current + n_lanes);

      auto out_current = out_scalar +  2 * idx_out_negative;
      hn::StoreU(hn::Mul(hn::Mul(v_in_1, v_prefac), v_ker_1), d, out_current);
      hn::StoreU(hn::Mul(hn::Mul(v_in_2, v_prefac), v_ker_2), d,
                 out_current + n_lanes);
    }

    for (; i < n_negative; ++i, ++idx_out_negative) {
      out[idx_out_negative] =
          prefac * in[i + offset_negative] * ker[n_negative - i];
    }
  }
}

template <typename T>
HWY_ATTR void downsample_2d_kernel(NeonufftModeOrder order, T prefac,
                                   std::array<IntType, 2> n_large,
                                   ConstHostView<std::complex<T>, 2> in,
                                   std::array<IntType, 2> n_small,
                                   std::array<const T *, 2> ker,
                                   HostView<std::complex<T>, 2> out) {

  const IntType n_negative = n_small[1] / 2; // [-N/2, -1]
  const IntType n_non_negative = n_small[1] - n_negative;// [0, N/2 -1]

  IntType idx_out_non_negative = n_negative;
  IntType idx_out_negative = 0;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_out_non_negative = 0;
    idx_out_negative = n_non_negative;
  }

  for (IntType i = 0; i < n_non_negative; ++i, ++idx_out_non_negative) {
    downsample_1d_kernel<T>(order, prefac * ker[1][i], n_large[0],
                                in.slice_view(i).data(), n_small[0], ker[0],
                                out.slice_view(idx_out_non_negative).data());
  }

  const IntType offset_negative = n_large[1] - n_negative;

  for (IntType i = 0; i < n_negative; ++i, ++idx_out_negative) {
    downsample_1d_kernel<T>(order, prefac * ker[1][n_negative - i], n_large[0],
                            in.slice_view(i + offset_negative).data(),
                            n_small[0], ker[0],
                            out.slice_view(idx_out_negative).data());
  }
}

template <typename T>
HWY_ATTR void downsample_3d_kernel(NeonufftModeOrder order,
                                   std::array<IntType, 3> n_large,
                                   ConstHostView<std::complex<T>, 3> in,
                                   std::array<IntType, 3> n_small,
                                   std::array<const T *, 3> ker,
                                   HostView<std::complex<T>, 3> out) {

  const IntType n_negative = n_small[2] / 2; // [-N/2, -1]
  const IntType n_non_negative = n_small[2] - n_negative;// [0, N/2 -1]

  IntType idx_out_non_negative = n_negative;
  IntType idx_out_negative = 0;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_out_non_negative = 0;
    idx_out_negative = n_non_negative;
  }

  for (IntType i = 0; i < n_non_negative; ++i, ++idx_out_non_negative) {
    downsample_2d_kernel<T>(order, ker[2][i], {n_large[0], n_large[1]},
                            in.slice_view(i), {n_small[0], n_small[1]},
                            {ker[0], ker[1]},
                            out.slice_view(idx_out_non_negative));
  }

  const IntType offset_negative = n_large[1] - n_negative;

  for (IntType i = 0; i < n_negative; ++i, ++idx_out_negative) {
    downsample_2d_kernel<T>(
        order, ker[2][n_negative - i], {n_large[0], n_large[1]},
        in.slice_view(i + offset_negative), {n_small[0], n_small[1]},
        {ker[0], ker[1]}, out.slice_view(idx_out_negative));
  }
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void downsample(NeonufftModeOrder order, std::array<IntType, DIM> n_large,
                ConstHostView<std::complex<T>, DIM> in,
                std::array<IntType, DIM> n_small,
                std::array<const T *, DIM> ker,
                HostView<std::complex<T>, DIM> out) {
  if constexpr (DIM == 1) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(downsample_1d_kernel<T>)
    (order, T(1), n_large[0], in.data(), n_small[0], ker[0], out.data());
  } else if constexpr (DIM == 2) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(downsample_2d_kernel<T>)
    (order, T(1), n_large, in, n_small, ker, out);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(downsample_3d_kernel<T>)
    (order, n_large, in, n_small, ker, out);
  }
}

template void downsample<float, 1>(NeonufftModeOrder order, 
                                   std::array<IntType, 1> n_large,
                                   ConstHostView<std::complex<float>, 1> in,
                                   std::array<IntType, 1> n_small,
                                   std::array<const float *, 1> ker,
                                   HostView<std::complex<float>, 1> out);

template void downsample<float, 2>(NeonufftModeOrder order, 
                                   std::array<IntType, 2> n_large,
                                   ConstHostView<std::complex<float>, 2> in,
                                   std::array<IntType, 2> n_small,
                                   std::array<const float *, 2> ker,
                                   HostView<std::complex<float>, 2> out);

template void downsample<float, 3>(NeonufftModeOrder order,
                                   std::array<IntType, 3> n_large,
                                   ConstHostView<std::complex<float>, 3> in,
                                   std::array<IntType, 3> n_small,
                                   std::array<const float *, 3> ker,
                                   HostView<std::complex<float>, 3> out);

template void downsample<double, 1>(NeonufftModeOrder order,
                                   std::array<IntType, 1> n_large,
                                   ConstHostView<std::complex<double>, 1> in,
                                   std::array<IntType, 1> n_small,
                                   std::array<const double *, 1> ker,
                                   HostView<std::complex<double>, 1> out);

template void downsample<double, 2>(NeonufftModeOrder order,
                                   std::array<IntType, 2> n_large,
                                   ConstHostView<std::complex<double>, 2> in,
                                   std::array<IntType, 2> n_small,
                                   std::array<const double *, 2> ker,
                                   HostView<std::complex<double>, 2> out);

template void downsample<double, 3>(NeonufftModeOrder order,
                                    std::array<IntType, 3> n_large,
                                    ConstHostView<std::complex<double>, 3> in,
                                    std::array<IntType, 3> n_small,
                                    std::array<const double *, 3> ker,
                                    HostView<std::complex<double>, 3> out);

#endif

} // namespace neonufft
