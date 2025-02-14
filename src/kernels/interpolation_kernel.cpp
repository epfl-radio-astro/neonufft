#include <cmath>
#include <complex>
#include <type_traits>

#include "neonufft/config.h"

#include "es_kernel_param.hpp"
#include "kernels/interpolation_kernel.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "neonufft/exceptions.hpp"
#include "neonufft/types.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/interpolation_kernel.cpp" // this file

#include "kernels/hwy_dispatch.hpp"

// must be included after highway headers
#include "kernels/es_kernel_eval.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T, typename D, typename V>
HWY_ATTR HWY_INLINE std::complex<T> reduce_cpx(D d, V v_sum) {
  constexpr IntType n_lanes = hn::Lanes(d);
  T out_real = 0;
  T out_imag = 0;
  for (IntType idx_lane = 0; idx_lane < n_lanes ; idx_lane += 2) {
    out_real += hn::ExtractLane(v_sum, idx_lane);
    out_imag += hn::ExtractLane(v_sum, idx_lane + 1);
  }

  return std::complex<T>{out_real, out_imag};
}

// Interpolate in 1d from "grid" using "ker" values. "ker" must be aligned.
// Both grid and ker must be 0-padded.
template <typename T, IntType N_SPREAD, typename D, typename V>
HWY_ATTR HWY_INLINE void
interp_inner_kernel(D d, const std::complex<T> *HWY_RESTRICT grid,
                    const T *HWY_RESTRICT ker, IntType idx_init,
                    IntType grid_size, V &v_sum_1) {
  // no wrapping, most common case
  const T *HWY_RESTRICT grid_s = reinterpret_cast<const T *>(grid);

  // compensate for complex -> scalar
  IntType idx_grid = 2 * idx_init;

  constexpr IntType n_lanes = hn::Lanes(d);

  // Iterate to 2 * N_SPREAD because of scalar iteration
  // Make use of 0-padding of kernel and grid
  for (IntType idx = 0; idx < 2 * N_SPREAD;
       idx += n_lanes, idx_grid += n_lanes) {
    auto v_grid_1 = hn::LoadU(d, grid_s + idx_grid);
    auto v_ker_1 = hn::Load(d, ker + idx); // ker is aligned
    v_sum_1 = hn::MulAdd(v_grid_1, v_ker_1, v_sum_1);
  }
}

// Interpolate in 1d from "grid" using "ker" values with wrapping.
template <typename T, IntType N_SPREAD>
HWY_ATTR HWY_INLINE std::complex<T>
interp_wrap_kernel(const std::complex<T> *HWY_RESTRICT grid,
                   const T *HWY_RESTRICT ker, IntType idx_init,
                   IntType grid_size) {
  std::complex<T> out = {0.0, 0.0};

  const T *HWY_RESTRICT grid_scalar = reinterpret_cast<const T *>(grid);
  IntType idx_ker = 0;
  IntType idx_grid = idx_init;

  T out_real = 0;
  T out_imag = 0;

  // wrap left
  for (; idx_ker < N_SPREAD && idx_grid < 0; ++idx_ker, ++idx_grid) {
    out_real = std::fma(grid_scalar[2 * (grid_size + idx_grid)],
                        ker[2 * idx_ker], out_real);
    out_imag = std::fma(grid_scalar[2 * (grid_size + idx_grid) + 1],
                        ker[2 * idx_ker + 1], out_imag);
  }

  // inner
  for (; idx_ker < N_SPREAD && idx_grid < grid_size; ++idx_ker, ++idx_grid) {
    out_real = std::fma(grid_scalar[2 * idx_grid], ker[2 * idx_ker], out_real);
    out_imag =
        std::fma(grid_scalar[2 * idx_grid + 1], ker[2 * idx_ker + 1], out_imag);
  }

  // wrap right
  for (; idx_ker < N_SPREAD; ++idx_ker, ++idx_grid) {
    out_real = std::fma(grid_scalar[2 * (idx_grid - grid_size)],
                        ker[2 * idx_ker], out_real);
    out_imag = std::fma(grid_scalar[2 * (idx_grid - grid_size) + 1],
                        ker[2 * idx_ker + 1], out_imag);
  }

  return {out_real, out_imag};
}

template <IntType DIM, typename KER, typename T>
HWY_ATTR void interpolation_kernel(const KER &kernel,
                                   ConstHostView<std::complex<T>, DIM> grid,
                                   IntType n_out,
                                   const Point<T, DIM> *HWY_RESTRICT px,
                                   std::complex<T> *HWY_RESTRICT out) {

  constexpr auto N_SPREAD = KER::N_SPREAD;
  const TagType<T> d;
  const T half_width = kernel.param.es_halfwidth; // half spread width
  constexpr IntType n_lanes = hn::Lanes(d);
  constexpr IntType lanes_padding = (N_SPREAD % n_lanes) + n_lanes;

  // Padded and aligned kernel values.
  // The first dim is stored in pairs of duplicates for easier loading.
  // The second and third dim is only size N_SPREAD.
  // We keep it in the same buffer to use only one allocation.
  // kx0, kx0, kx1, kx1, ...
  // ky0, ky1, ky2, ky3, ...
  // kz0, kz1, kz2, kz3, ...
  constexpr IntType ker_size_padded_y = DIM > 1 ? N_SPREAD + lanes_padding : 0;
  constexpr IntType ker_size_padded_z = DIM > 2 ? N_SPREAD + lanes_padding : 0;
  HWY_ALIGN
  std::array<T, 2 * (N_SPREAD + lanes_padding)> ker_values_x = {
      0}; // must be 0-padded
  HWY_ALIGN std::array<T, ker_size_padded_y> ker_values_y;
  HWY_ALIGN std::array<T, ker_size_padded_z> ker_values_z;

  std::array<T, DIM> x;
  std::array<IntType, DIM> idx_init;
  std::array<IntType, DIM> grid_size;
  if constexpr (DIM == 1) {
    grid_size[0] = grid.shape(0);
  } else {
    grid_size = grid.shape();
  }

  // loop over nu points
  for (IntType idx_nu = 0; idx_nu < n_out; idx_nu += 1) {
    const auto point = px[idx_nu];
    for (IntType dim = 0; dim < DIM; ++dim) {
      // NOTE: slight numerical difference here to finufft around integer
      // numbers
      const T loc = point.coord[dim] * grid_size[dim]; // px in [0, 1]
      const T idx_init_ceil = std::ceil(loc - half_width);
      idx_init[dim] = IntType(idx_init_ceil); // fine padded_grid start index (-
      x[dim] = idx_init_ceil - loc; // x1 in [-w/2,-w/2+1], up to rounding
    }

    // evaluate kernel, store in padded buffer
    kernel.eval2(d, x[0], ker_values_x.data());
    if constexpr (DIM > 1) {
      kernel.eval(d, x[1], ker_values_y.data());
    }
    if constexpr (DIM > 2) {
      kernel.eval(d, x[2], ker_values_z.data());
    }

    if constexpr (DIM == 1) {
      if (idx_init[0] < 0 || idx_init[0] + N_SPREAD >= grid_size[0]) {
        out[point.index] = interp_wrap_kernel<T, N_SPREAD>(
            &grid[0], ker_values_x.data(), idx_init[0], grid_size[0]);
      } else {
        auto v_sum_1 = hn::Zero(d);

        interp_inner_kernel<T, N_SPREAD>(d, &grid[0], ker_values_x.data(),
                                         idx_init[0], grid_size[0], v_sum_1);
        out[point.index] = reduce_cpx<T>(d, v_sum_1);
      }
    } else if constexpr (DIM == 2) {
      if (idx_init[0] < 0 || idx_init[0] + N_SPREAD >= grid_size[0] ||
          idx_init[1] < 0 || idx_init[1] + N_SPREAD >= grid_size[1]) {
        std::complex<T> out_val = {0, 0};
        for (IntType idx_ker_y = 0; idx_ker_y < N_SPREAD; ++idx_ker_y) {
          IntType idx_y = idx_ker_y + idx_init[1];
          if (idx_y < 0)
            idx_y += grid_size[1];
          if (idx_y >= grid_size[1])
            idx_y -= grid_size[1];

          const auto ker_val_y = ker_values_y[idx_ker_y];

          out_val += ker_val_y * interp_wrap_kernel<T, N_SPREAD>(
                                     &grid[{0, idx_y}], ker_values_x.data(),
                                     idx_init[0], grid_size[0]);
        }
        out[point.index] = out_val;

      } else {

        auto v_sum_1 = hn::Zero(d);
        for (IntType idx_ker_y = 0; idx_ker_y < N_SPREAD; ++idx_ker_y) {
          const IntType idx_y = idx_ker_y + idx_init[1];

          const auto ker_val_y = hn::Set(d, ker_values_y[idx_ker_y]);

          auto v_res_1 = hn::Zero(d);
          interp_inner_kernel<T, N_SPREAD>(d, &grid[{0, idx_y}],
                                           ker_values_x.data(), idx_init[0],
                                           grid_size[0], v_res_1);
          v_sum_1 = hn::MulAdd(ker_val_y, v_res_1, v_sum_1);
        }
        out[point.index] = reduce_cpx<T>(d, v_sum_1);
      }
    } else {
      if (idx_init[0] < 0 || idx_init[0] + N_SPREAD >= grid_size[0] ||
          idx_init[1] < 0 || idx_init[1] + N_SPREAD >= grid_size[1] ||
          idx_init[2] < 0 || idx_init[2] + N_SPREAD >= grid_size[2]) {
        std::complex<T> out_val = {0, 0};
        for (IntType idx_ker_z = 0; idx_ker_z < N_SPREAD; ++idx_ker_z) {
          IntType idx_z = idx_ker_z + idx_init[2];
          if (idx_z < 0)
            idx_z += grid_size[2];
          if (idx_z >= grid_size[2])
            idx_z -= grid_size[2];

          const auto ker_val_z = ker_values_z[idx_ker_z];

          for (IntType idx_ker_y = 0; idx_ker_y < N_SPREAD; ++idx_ker_y) {
            IntType idx_y = idx_ker_y + idx_init[1];
            if (idx_y < 0)
              idx_y += grid_size[1];
            if (idx_y >= grid_size[1])
              idx_y -= grid_size[1];

            const auto ker_val_y = ker_values_y[idx_ker_y];

            out_val += (ker_val_y * ker_val_z) *
                       interp_wrap_kernel<T, N_SPREAD>(
                           &grid[{0, idx_y, idx_z}], ker_values_x.data(),
                           idx_init[0], grid_size[0]);
          }
        }
        out[point.index] = out_val;
      } else {
        auto v_sum_1 = hn::Zero(d);
        for (IntType idx_ker_z = 0; idx_ker_z < N_SPREAD; ++idx_ker_z) {
          const IntType idx_z = idx_ker_z + idx_init[2];

          const auto ker_val_z = hn::Set(d, ker_values_z[idx_ker_z]);

          for (IntType idx_ker_y = 0; idx_ker_y < N_SPREAD; ++idx_ker_y) {
            const IntType idx_y = idx_ker_y + idx_init[1];

            const auto ker_val_yz =
                hn::Mul(hn::Set(d, ker_values_y[idx_ker_y]), ker_val_z);

            auto v_res_1 = hn::Zero(d);
            interp_inner_kernel<T, N_SPREAD>(d, &grid[{0, idx_y, idx_z}],
                                             ker_values_x.data(), idx_init[0],
                                             grid_size[0], v_res_1);
            v_sum_1 = hn::MulAdd(ker_val_yz, v_res_1, v_sum_1);
          }
        }

        out[point.index] = reduce_cpx<T>(d, v_sum_1);
      }
    }
  }
}

template <IntType DIM, typename T, IntType N_SPREAD>
HWY_ATTR void interpolate_dispatch(NeonufftKernelType kernel_type,
                                      const KernelParameters<T> &kernel_param,
                                      ConstHostView<std::complex<T>, DIM> grid,
                                      IntType n_out, const Point<T, DIM> *px,
                                      std::complex<T> *out) {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  if (kernel_param.n_spread == N_SPREAD) {
    if (kernel_type == NEONUFFT_ES_KERNEL) {
      if(kernel_param.approximation && kernel_param.upsampfac == 2.0) {
        EsKernelHorner200<T, N_SPREAD> kernel(kernel_param);
        interpolation_kernel<DIM, decltype(kernel), T>(kernel, grid, n_out, px,
                                                       out);
      } else if (kernel_param.approximation && kernel_param.upsampfac == 1.25) {
        EsKernelHorner125<T, N_SPREAD> kernel(kernel_param);
        interpolation_kernel<DIM, decltype(kernel), T>(kernel, grid, n_out, px,
                                                       out);
      } else {
        EsKernelDirect<T, N_SPREAD> kernel(kernel_param);
        interpolation_kernel<DIM, decltype(kernel), T>(kernel, grid, n_out, px,
                                                       out);
      }
    } else {
      throw InternalError("Unknown kernel type");
    }
  } else {
    if constexpr (N_SPREAD > 2) {
      interpolate_dispatch<DIM, T, N_SPREAD - 1>(kernel_type, kernel_param,
                                                    grid, n_out, px, out);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}


template <IntType DIM>
HWY_ATTR void interpolate_float(NeonufftKernelType kernel_type,
                                   const KernelParameters<float> &kernel_param,
                                   ConstHostView<std::complex<float>, DIM> grid,
                                   IntType n_out, const Point<float, DIM> *px,
                                   std::complex<float> *out) {
  interpolate_dispatch<DIM, float, 16>(kernel_type, kernel_param, grid,
                                          n_out, px, out);
}

template <IntType DIM>
HWY_ATTR void
interpolate_double(NeonufftKernelType kernel_type,
                      const KernelParameters<double> &kernel_param,
                      ConstHostView<std::complex<double>, DIM> grid,
                      IntType n_out, const Point<double, DIM> *px,
                      std::complex<double> *out) {
  interpolate_dispatch<DIM, double, 16>(kernel_type, kernel_param, grid,
                                           n_out, px, out);
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void interpolate(NeonufftKernelType kernel_type,
                    const KernelParameters<T> &kernel_param,
                    ConstHostView<std::complex<T>, DIM> grid, IntType n_out,
                    const Point<T, DIM> *px, std::complex<T> *out) {
  if constexpr (std::is_same_v<T, float>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(interpolate_float<DIM>)
    (kernel_type, kernel_param, grid, n_out, px, out);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(interpolate_double<DIM>)
    (kernel_type, kernel_param, grid, n_out, px, out);
  }
}

template void interpolate<float, 1>(
    NeonufftKernelType kernel_type, const KernelParameters<float> &kernel_param,
    ConstHostView<std::complex<float>, 1> grid, IntType n_out,
    const Point<float, 1> *px, std::complex<float> *out);

template void interpolate<float, 2>(
    NeonufftKernelType kernel_type, const KernelParameters<float> &kernel_param,
    ConstHostView<std::complex<float>, 2> grid, IntType n_out,
    const Point<float, 2> *px, std::complex<float> *out);

template void interpolate<float, 3>(
    NeonufftKernelType kernel_type, const KernelParameters<float> &kernel_param,
    ConstHostView<std::complex<float>, 3> grid, IntType n_out,
    const Point<float, 3> *px, std::complex<float> *out);

template void interpolate<double, 1>(
    NeonufftKernelType kernel_type, const KernelParameters<double> &kernel_param,
    ConstHostView<std::complex<double>, 1> grid, IntType n_out,
    const Point<double, 1> *px, std::complex<double> *out);

template void interpolate<double, 2>(
    NeonufftKernelType kernel_type, const KernelParameters<double> &kernel_param,
    ConstHostView<std::complex<double>, 2> grid, IntType n_out,
    const Point<double, 2> *px, std::complex<double> *out);

template void interpolate<double, 3>(
    NeonufftKernelType kernel_type, const KernelParameters<double> &kernel_param,
    ConstHostView<std::complex<double>, 3> grid, IntType n_out,
    const Point<double, 3> *px, std::complex<double> *out);

#endif

} // namespace neonufft
