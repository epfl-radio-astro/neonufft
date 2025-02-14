#include <cassert>
#include <complex>
#include <cstring>

#include "neonufft/config.h"
#include "neonufft/enums.h"

#include "es_kernel_param.hpp"
#include "kernels/fold_padding.hpp"
#include "memory/view.hpp"
#include "neonufft/types.hpp"
#include "util/spread_padding.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/fold_padding.cpp" // this file

#include "kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR void fold_padding_1d_kernel(IntType n_spread,
                                     HostView<std::complex<T>, 1> padded_grid) {
  const IntType padding = spread_padding(n_spread);

  const IntType grid_size = padded_grid.shape(0) - 2 * padding;
  // left
  {
    const T *HWY_RESTRICT ptr_src =
        reinterpret_cast<const T *>(&padded_grid[0]);
    T *HWY_RESTRICT ptr_tgt = reinterpret_cast<T *>(&padded_grid[grid_size]);
    for (IntType idx_x = 0; idx_x < 2 * padding; ++idx_x) {
      ptr_tgt[idx_x] += ptr_src[idx_x];
    }
  }

  // right
  {
    const T *HWY_RESTRICT ptr_src =
        reinterpret_cast<const T *>(&padded_grid[padding + grid_size]);
    T *HWY_RESTRICT ptr_tgt = reinterpret_cast<T *>(&padded_grid[padding]);
    for (IntType idx_x = 0; idx_x < 2 * padding; ++idx_x) {
      ptr_tgt[idx_x] += ptr_src[idx_x];
    }
  }
}

template <typename T>
HWY_ATTR void fold_padding_2d_kernel(IntType n_spread,
                                     HostView<std::complex<T>, 2> padded_grid) {

  const IntType padding = spread_padding(n_spread);

  std::array<IntType, 2> grid_size = {padded_grid.shape(0) - 2 * padding,
                                      padded_grid.shape(1) - 2 * padding};

  // fold top
  for (IntType idx_y = 0; idx_y < padding; ++idx_y) {
    const T *HWY_RESTRICT ptr_src =
        reinterpret_cast<const T *>(&padded_grid[{0, idx_y}]);
    T *HWY_RESTRICT ptr_tgt =
        reinterpret_cast<T *>(&padded_grid[{0, grid_size[1] + idx_y}]);

    for (IntType idx_x = 0; idx_x < 2 * padded_grid.shape(0); ++idx_x) {
      ptr_tgt[idx_x] += ptr_src[idx_x];
    }
  }

  // fold bottom
  for (IntType idx_y = 0; idx_y < padding; ++idx_y) {
    const T *HWY_RESTRICT ptr_src = reinterpret_cast<const T *>(
        &padded_grid[{0, padding + grid_size[1] + idx_y}]);
    T *HWY_RESTRICT ptr_tgt =
        reinterpret_cast<T *>(&padded_grid[{0, padding + idx_y}]);

    for (IntType idx_x = 0; idx_x < 2 * padded_grid.shape(0); ++idx_x) {
      ptr_tgt[idx_x] += ptr_src[idx_x];
    }
  }

  // fold left / right
  for (IntType idx_y = padding; idx_y < padding + grid_size[1]; ++idx_y) {
    fold_padding_1d_kernel(n_spread, padded_grid.slice_view(idx_y));
  }
}

template <typename T>
HWY_ATTR void fold_padding_3d_kernel(IntType n_spread,
                                     HostView<std::complex<T>, 3> padded_grid) {

  const IntType padding = spread_padding(n_spread);

  std::array<IntType, 3> grid_size = {padded_grid.shape(0) - 2 * padding,
                                      padded_grid.shape(1) - 2 * padding,
                                      padded_grid.shape(2) - 2 * padding};

  // fold top
  for (IntType idx_z = 0; idx_z < padding; ++idx_z) {
    auto src_slice_z = padded_grid.slice_view(idx_z);
    auto tgt_slice_z = padded_grid.slice_view(grid_size[2] + idx_z);
    for (IntType idx_y = 0; idx_y < padded_grid.shape(1); ++idx_y) {
      const T *HWY_RESTRICT ptr_src =
          reinterpret_cast<const T *>(&src_slice_z[{0, idx_y}]);
      T *HWY_RESTRICT ptr_tgt = reinterpret_cast<T *>(&tgt_slice_z[{0, idx_y}]);
      for (IntType idx_x = 0; idx_x < 2 * src_slice_z.shape(0); ++idx_x) {
        ptr_tgt[idx_x] += ptr_src[idx_x];
      }
    }
  }

  // fold bottom
  for (IntType idx_z = 0; idx_z < padding; ++idx_z) {
    auto src_slice_z = padded_grid.slice_view(padding + grid_size[2] + idx_z);
    auto tgt_slice_z = padded_grid.slice_view(padding + idx_z);
    for (IntType idx_y = 0; idx_y < padded_grid.shape(1); ++idx_y) {
      const T *HWY_RESTRICT ptr_src =
          reinterpret_cast<const T *>(&src_slice_z[{0, idx_y}]);
      T *HWY_RESTRICT ptr_tgt = reinterpret_cast<T *>(&tgt_slice_z[{0, idx_y}]);
      for (IntType idx_x = 0; idx_x < 2 * src_slice_z.shape(0); ++idx_x) {
        ptr_tgt[idx_x] += ptr_src[idx_x];
      }
    }
  }

  // fold left / right
  for (IntType idx_z = padding; idx_z < padding + grid_size[2]; ++idx_z) {
    fold_padding_2d_kernel(n_spread, padded_grid.slice_view(idx_z));
  }
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE
template <typename T, IntType DIM>
void fold_padding(IntType n_spread,
                  HostView<std::complex<T>, DIM> padded_grid) {
  if constexpr (DIM == 1) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(fold_padding_1d_kernel<T>)
    (n_spread, padded_grid);
  } else if constexpr (DIM == 2) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(fold_padding_2d_kernel<T>)
    (n_spread, padded_grid);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(fold_padding_3d_kernel<T>)
    (n_spread, padded_grid);
  }
}

template void
fold_padding<float, 1>(IntType n_spread,
                       HostView<std::complex<float>, 1> padded_grid);

template void
fold_padding<float, 2>(IntType n_spread,
                       HostView<std::complex<float>, 2> padded_grid);

template void
fold_padding<float, 3>(IntType n_spread,
                       HostView<std::complex<float>, 3> padded_grid);

template void
fold_padding<double, 1>(IntType n_spread,
                        HostView<std::complex<double>, 1> padded_grid);

template void
fold_padding<double, 2>(IntType n_spread,
                        HostView<std::complex<double>, 2> padded_grid);

template void
fold_padding<double, 3>(IntType n_spread,
                        HostView<std::complex<double>, 3> padded_grid);
#endif

} // namespace neonufft
