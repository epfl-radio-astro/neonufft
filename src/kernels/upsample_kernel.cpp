#include <cassert>
#include <complex>
#include <cstring>

#include "neonufft/config.h"
#include "neonufft/enums.h"

#include "es_kernel_param.hpp"
#include "kernels/upsample_kernel.hpp"
#include "memory/view.hpp"
#include "neonufft/types.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/upsample_kernel.cpp" // this file

#include "kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR void upsample_1d_kernel(NeonufftModeOrder order, T prefac,
                                 IntType n_small, const T *HWY_RESTRICT ker,
                                 const std::complex<T> *HWY_RESTRICT in,
                                 IntType n_large,
                                 std::complex<T> *HWY_RESTRICT out) {
  assert(n_small <= n_large);

  const TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);

  const auto v_prefac = hn::Set(d, prefac);

  const IntType n_negative = n_small / 2; // [-N/2, -1]
  const IntType n_non_negative = n_small - n_negative;// [0, N/2 -1]

  IntType idx_in_non_negative = n_negative;
  IntType idx_in_negative = 0;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_in_non_negative = 0;
    idx_in_negative = n_non_negative;
  }

  auto in_scalar = reinterpret_cast<const T *>(in);
  auto out_scalar = reinterpret_cast<T *>(out);

  {
    IntType idx = 0;
    for (; idx + n_lanes <= n_non_negative;
         idx += n_lanes, idx_in_non_negative += n_lanes) {
      // duplicate each ker value to multiply with real and imag interleaved
      auto v_ker = hn::Load(d, ker + idx);
      auto v_ker_1 = hn::InterleaveWholeLower(d, v_ker, v_ker);
      auto v_ker_2 = hn::InterleaveWholeUpper(d, v_ker, v_ker);

      // load complex values as scalars, since we only multiply with real values
      auto in_current = in_scalar + 2 * idx_in_non_negative;
      auto v_in_1 = hn::LoadU(d, in_current);
      auto v_in_2 = hn::LoadU(d, in_current + n_lanes);

      auto out_current = out_scalar + 2 * idx;
      // the output grid might not be aligned if padded
      hn::StoreU(hn::Mul(hn::Mul(v_in_1, v_prefac), v_ker_1), d, out_current);
      hn::StoreU(hn::Mul(hn::Mul(v_in_2, v_prefac), v_ker_2), d,
                 out_current + n_lanes);
    }

    for (; idx < n_non_negative; ++idx, ++idx_in_non_negative) {
      out[idx] = prefac * in[idx_in_non_negative] * ker[idx];
    }
  }

  const IntType offset_negative = n_large - n_negative;

  {
    IntType idx = 0;
    for (; idx + n_lanes <= n_negative;
         idx += n_lanes, idx_in_negative += n_lanes) {
      // duplicate each ker value to multiply with real and imag interleaved
      // reverse because ker is read backwards
      auto v_ker =
          hn::Reverse(d, hn::LoadU(d, ker + n_negative - idx - n_lanes + 1));
      auto v_ker_1 = hn::InterleaveWholeLower(d, v_ker, v_ker);
      auto v_ker_2 = hn::InterleaveWholeUpper(d, v_ker, v_ker);

      // load complex values as scalars, since we only multiply with real values
      auto in_current = in_scalar + 2 * idx_in_negative;
      auto v_in_1 = hn::LoadU(d, in_current);
      auto v_in_2 = hn::LoadU(d, in_current + n_lanes);

      auto out_current = out_scalar + 2 * (idx + offset_negative);
      hn::StoreU(hn::Mul(hn::Mul(v_in_1, v_prefac), v_ker_1), d, out_current);
      hn::StoreU(hn::Mul(hn::Mul(v_in_2, v_prefac), v_ker_2), d,
                 out_current + n_lanes);
    }

    for (; idx < n_negative; ++idx, ++idx_in_negative) {
      out[idx + offset_negative] =
          prefac * in[idx_in_negative] * ker[n_negative  - idx];
    }
  }
}

template <typename T>
HWY_ATTR void upsample_2d_kernel(NeonufftModeOrder order, T prefac,
                                 ConstHostView<std::complex<T>, 2> small_grid,
                                 std::array<ConstHostView<T, 1>, 2> ker,
                                 HostView<std::complex<T>, 2> large_grid) {

  const auto large_grid_size = large_grid.shape();
  const auto small_grid_size = small_grid.shape();

  assert(small_grid_size[1] <= large_grid_size[1]);

  const IntType n_negative = small_grid_size[1] / 2;              // [-N/2, -1]
  const IntType n_non_negative = small_grid_size[1] - n_negative; // [0, N/2 -1]
  const IntType padding = large_grid_size[1] - small_grid_size[1];

  IntType idx_in_non_negative = n_negative;
  IntType idx_in_negative = 0;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_in_non_negative = 0;
    idx_in_negative = n_non_negative;
  }

  const T *HWY_RESTRICT local_ker_ptr = ker[1].data();
  for (IntType idx = 0; idx < n_non_negative; ++idx, ++idx_in_non_negative) {
    const auto s = prefac * local_ker_ptr[idx];
    upsample_1d_kernel<T>(order, s, small_grid_size[0], ker[0].data(),
                          &small_grid[{0, idx_in_non_negative}],
                          large_grid_size[0], &large_grid[{0, idx}]);
  }

  for (IntType idx = 0; idx < n_negative; ++idx, ++idx_in_negative) {
    const auto s = prefac * local_ker_ptr[n_negative - idx];

    upsample_1d_kernel<T>(order, s, small_grid_size[0], ker[0].data(),
                          &small_grid[{0, idx_in_negative}], large_grid_size[0],
                          &large_grid[{0, n_non_negative + padding + idx}]);
  }
}

template <typename T>
HWY_ATTR void upsample_3d_kernel(NeonufftModeOrder order,
                                 ConstHostView<std::complex<T>, 3> small_grid,
                                 std::array<ConstHostView<T, 1>, 3> ker,
                                 HostView<std::complex<T>, 3> large_grid) {

  const auto large_grid_size = large_grid.shape();
  const auto small_grid_size = small_grid.shape();

  assert(small_grid_size[2] <= large_grid_size[2]);

  const IntType n_negative = small_grid_size[2] / 2;              // [-N/2, -1]
  const IntType n_non_negative = small_grid_size[2] - n_negative; // [0, N/2 -1]
  const IntType padding = large_grid_size[2] - small_grid_size[2];

  IntType idx_in_non_negative = n_negative;
  IntType idx_in_negative = 0;

  if (order == NEONUFFT_MODE_ORDER_FFT) {
    idx_in_non_negative = 0;
    idx_in_negative = n_non_negative;
  }

  const T *HWY_RESTRICT local_ker_ptr = ker[2].data();
  for (IntType idx = 0; idx < n_non_negative; ++idx, ++idx_in_non_negative) {
    const auto s = local_ker_ptr[idx];
    upsample_2d_kernel<T>(order, s, small_grid.slice_view(idx_in_non_negative),
                          {ker[0], ker[1]}, large_grid.slice_view(idx));
  }

  for (IntType idx = 0; idx < n_negative; ++idx, ++idx_in_negative) {
    const auto s = local_ker_ptr[n_negative - idx];

    upsample_2d_kernel<T>(
        order, s, small_grid.slice_view(idx_in_negative), {ker[0], ker[1]},
        large_grid.slice_view(n_non_negative + padding + idx));
  }
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void upsample(NeonufftModeOrder order,
              ConstHostView<std::complex<T>, DIM> small_grid,
              std::array<ConstHostView<T, 1>, DIM> ker,
              HostView<std::complex<T>, DIM> large_grid) {
  if constexpr (DIM == 1) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(upsample_1d_kernel<T>)
    (order, 1, small_grid.shape(0), ker[0].data(), small_grid.data(),
     large_grid.shape(0), large_grid.data());
  } else if constexpr(DIM == 2) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(upsample_2d_kernel<T>)
    (order, 1, small_grid, ker, large_grid);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(upsample_3d_kernel<T>)
    (order, small_grid, ker, large_grid);
  }
}

template void
upsample<float, 1>(NeonufftModeOrder order,
                   ConstHostView<std::complex<float>, 1> small_grid,
                   std::array<ConstHostView<float, 1>, 1> ker,
                   HostView<std::complex<float>, 1> large_grid);

template void
upsample<float, 2>(NeonufftModeOrder order,
                   ConstHostView<std::complex<float>, 2> small_grid,
                   std::array<ConstHostView<float, 1>, 2> ker,
                   HostView<std::complex<float>, 2> large_grid);

template void
upsample<float, 3>(NeonufftModeOrder order,
                   ConstHostView<std::complex<float>, 3> small_grid,
                   std::array<ConstHostView<float, 1>, 3> ker,
                   HostView<std::complex<float>, 3> large_grid);

template void
upsample<double, 1>(NeonufftModeOrder order,
                   ConstHostView<std::complex<double>, 1> small_grid,
                   std::array<ConstHostView<double, 1>, 1> ker,
                   HostView<std::complex<double>, 1> large_grid);

template void
upsample<double, 2>(NeonufftModeOrder order,
                   ConstHostView<std::complex<double>, 2> small_grid,
                   std::array<ConstHostView<double, 1>, 2> ker,
                   HostView<std::complex<double>, 2> large_grid);

template void
upsample<double, 3>(NeonufftModeOrder order,
                    ConstHostView<std::complex<double>, 3> small_grid,
                    std::array<ConstHostView<double, 1>, 3> ker,
                    HostView<std::complex<double>, 3> large_grid);
#endif

} // namespace neonufft
