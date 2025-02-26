#include <cmath>
#include <complex>
#include <cstring>

#include "neonufft/config.h"

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/kernels/spreading_kernel.hpp"
#include "neonufft/memory/array.hpp"
#include "neonufft/types.hpp"
#include "neonufft/util/point.hpp"
#include "neonufft/util/spread_padding.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "neonufft/kernels/spreading_kernel.cpp" // this file

#include "neonufft/kernels/hwy_dispatch.hpp"

// must be included after highway headers
#include "neonufft/kernels/es_kernel_eval.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <IntType N_SPREAD, typename T, typename D, typename V>
HWY_ATTR HWY_INLINE void
spread_inner_kernel(D d, IntType idx_out, V v_in_val,
                    const T *HWY_RESTRICT ker_aligned,
                    std::complex<T> *HWY_RESTRICT padded_grid) {

  T *HWY_RESTRICT grid_scalar = reinterpret_cast<T *>(padded_grid);
  constexpr IntType n_lanes = hn::Lanes(d);

  idx_out *= 2; // compensate for complex -> scalar

  // Iterate to 2 * N_SPREAD because of scalar iteration
  // Make use of 0-padding of kernel and padded_grid
  for (IntType idx_ker = 0; idx_ker < 2 * N_SPREAD;
       idx_ker += n_lanes, idx_out += n_lanes) {

    // ker is aligned
    const auto v_ker_1 = hn::Load(d, ker_aligned + idx_ker);

    auto v_g1 = hn::LoadU(d, grid_scalar + idx_out);

    hn::StoreU(hn::MulAdd(v_in_val, v_ker_1, v_g1), d, grid_scalar + idx_out);
  }
}

template <IntType DIM, typename KER, typename T>
HWY_ATTR void
spread_kernel(const KER &kernel, IntType num_nu,
              const Point<T, DIM> *HWY_RESTRICT px,
              const std::complex<T> *HWY_RESTRICT input,
              const std::complex<T> *HWY_RESTRICT prephase_optional,
              std::array<IntType, DIM> grid_size,
              HostView<std::complex<T>, DIM> padded_grid) {

  constexpr IntType N_SPREAD = KER::N_SPREAD;

  const TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);
  constexpr IntType lanes_padding = (N_SPREAD % n_lanes) + n_lanes;

  const T half_width = kernel.param.es_halfwidth; // half spread width

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

  // loop over nu points
  for (IntType idx_nu = 0; idx_nu < num_nu; ++idx_nu) {
    const auto point = px[idx_nu];

    for (IntType dim = 0; dim < DIM; ++dim) {
      // NOTE: slight numerical difference here to finufft around integer
      // numbers
      const T loc = point.coord[dim] * grid_size[dim]; // px in [0, 1]
      const T idx_init_ceil = std::ceil(loc - half_width);
      idx_init[dim] = IntType(idx_init_ceil) +
                      spread_padding(N_SPREAD); // fine padded_grid start index (-
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

    auto in_val = input[point.index];
    // optionally apply prephase to input. Prephase must NOT be sorted.
    if (prephase_optional) {
      in_val *= prephase_optional[point.index];
    }

    // Add to upsampled padded_grid. The small padded_grid has order (negative, non-negative).
    // The large padded_grid has order (non-negative, negative).
    // idx_init / idx_out is in reference to small padded_grid.

    const auto v_in_val_real = hn::Set(d, in_val.real());
    const auto v_in_val_imag = hn::Set(d, in_val.imag());
    const auto v_in_val =
        hn::InterleaveWholeLower(d, v_in_val_real, v_in_val_imag);

    if constexpr (DIM == 1) {
      spread_inner_kernel<N_SPREAD, T>(d, idx_init[0], v_in_val,
                                       ker_values_x.data(), padded_grid.data());
    } else if constexpr (DIM == 2) {
      for (IntType idx_ker_y = 0; idx_ker_y < N_SPREAD; ++idx_ker_y) {
        const IntType idx_y = idx_ker_y + idx_init[1];

        const auto v_in_val_scaled =
            hn::Mul(v_in_val, hn::Set(d, ker_values_y[idx_ker_y]));

        spread_inner_kernel<N_SPREAD, T>(d, idx_init[0], v_in_val_scaled,
                                         ker_values_x.data(),
                                         &padded_grid[{0, idx_y}]);
      }

    } else {
      for (IntType idx_ker_z = 0; idx_ker_z < N_SPREAD; ++idx_ker_z) {
        const IntType idx_z = idx_ker_z + idx_init[2];

        const auto v_in_val_scaled_z =
            hn::Mul(v_in_val, hn::Set(d, ker_values_z[idx_ker_z]));

        for (IntType idx_ker_y = 0; idx_ker_y < N_SPREAD; ++idx_ker_y) {
          const IntType idx_y = idx_ker_y + idx_init[1];

          const auto v_in_val_scaled_yz =
              hn::Mul(v_in_val_scaled_z, hn::Set(d, ker_values_y[idx_ker_y]));

          spread_inner_kernel<N_SPREAD, T>(d, idx_init[0], v_in_val_scaled_yz,
                                           ker_values_x.data(),
                                           &padded_grid[{0, idx_y, idx_z}]);
        }
      }
    }
  }
}

template <IntType DIM, typename T, IntType N_SPREAD>
HWY_ATTR void spread_dispatch(NeonufftKernelType kernel_type,
                              const KernelParameters<T> &kernel_param,
                              IntType num_nu, const Point<T, DIM> *px,
                              const std::complex<T> *input,
                              const std::complex<T> *prephase_optional,std::array<IntType, DIM> grid_size,
                              HostView<std::complex<T>, DIM> padded_grid) {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  if (kernel_param.n_spread == N_SPREAD) {
    if (kernel_type == NEONUFFT_ES_KERNEL) {
      if(kernel_param.approximation && kernel_param.upsampfac == 2.0) {
        EsKernelHorner200<T, N_SPREAD> kernel(kernel_param);
        spread_kernel<DIM, decltype(kernel), T>(kernel, num_nu, px, input,
                                                prephase_optional, grid_size,
                                                padded_grid);
      } else if (kernel_param.approximation && kernel_param.upsampfac == 1.25) {
        EsKernelHorner125<T, N_SPREAD> kernel(kernel_param);
        spread_kernel<DIM, decltype(kernel), T>(kernel, num_nu, px, input,
                                                prephase_optional, grid_size,
                                                padded_grid);
      } else {
      EsKernelDirect<T, N_SPREAD> kernel(kernel_param);
      spread_kernel<DIM, decltype(kernel), T>(kernel, num_nu, px, input,
                                              prephase_optional,grid_size, padded_grid);
      }
    } else {
      throw InternalError("Unknown kernel type");
    }
  } else {
    if constexpr (N_SPREAD > 2) {
      spread_dispatch<DIM, T, N_SPREAD - 1>(kernel_type, kernel_param, num_nu,
                                            px, input, prephase_optional,
                                            grid_size, padded_grid);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}

template <IntType DIM>
HWY_ATTR void spread_float(NeonufftKernelType kernel_type,
                           const KernelParameters<float> &kernel_param,
                           IntType num_nu, const Point<float, DIM> *px,
                           const std::complex<float> *input,
                           const std::complex<float> *prephase_optional,std::array<IntType, DIM> grid_size,
                           HostView<std::complex<float>, DIM> padded_grid) {
  spread_dispatch<DIM, float, 16>(kernel_type, kernel_param, num_nu, px, input,
                                  prephase_optional,grid_size, padded_grid);
}

template <IntType DIM>
HWY_ATTR void spread_double(NeonufftKernelType kernel_type,
                            const KernelParameters<double> &kernel_param,
                            IntType num_nu, const Point<double, DIM> *px,
                            const std::complex<double> *input,
                            const std::complex<double> *prephase_optional,std::array<IntType, DIM> grid_size,
                            HostView<std::complex<double>, DIM> padded_grid) {
  spread_dispatch<DIM, double, 16>(kernel_type, kernel_param, num_nu, px, input,
                                   prephase_optional,grid_size, padded_grid);
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void spread(NeonufftKernelType kernel_type,
            const KernelParameters<T> &kernel_param, IntType num_nu,
            const Point<T, DIM> *px, const std::complex<T> *input,
            const std::complex<T> *prephase_optional,
            std::array<IntType, DIM> grid_size,
            HostView<std::complex<T>, DIM> padded_grid) {
  if constexpr (std::is_same_v<T, float>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(spread_float<DIM>)
    (kernel_type, kernel_param, num_nu, px, input, prephase_optional,grid_size, padded_grid);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(spread_double<DIM>)
    (kernel_type, kernel_param, num_nu, px, input, prephase_optional,grid_size, padded_grid);
  }
}

template void spread<float, 1>(NeonufftKernelType kernel_type,
                               const KernelParameters<float> &kernel_param,
                               IntType num_nu, const Point<float, 1> *px,
                               const std::complex<float> *input,
                               const std::complex<float> *prephase_optional,std::array<IntType, 1> grid_size,
                               HostView<std::complex<float>, 1> padded_grid);

template void spread<float, 2>(NeonufftKernelType kernel_type,
                               const KernelParameters<float> &kernel_param,
                               IntType num_nu, const Point<float, 2> *px,
                               const std::complex<float> *input,
                               const std::complex<float> *prephase_optional,std::array<IntType, 2> grid_size,
                               HostView<std::complex<float>, 2> padded_grid);

template void spread<float, 3>(NeonufftKernelType kernel_type,
                               const KernelParameters<float> &kernel_param,
                               IntType num_nu, const Point<float, 3> *px,
                               const std::complex<float> *input,
                               const std::complex<float> *prephase_optional,std::array<IntType, 3> grid_size,
                               HostView<std::complex<float>, 3> padded_grid);

template void spread<double, 1>(NeonufftKernelType kernel_type,
                                const KernelParameters<double> &kernel_param,
                                IntType num_nu, const Point<double, 1> *px,
                                const std::complex<double> *input,
                                const std::complex<double> *prephase_optional,std::array<IntType, 1> grid_size,
                                HostView<std::complex<double>, 1> padded_grid);

template void spread<double, 2>(NeonufftKernelType kernel_type,
                                const KernelParameters<double> &kernel_param,
                                IntType num_nu, const Point<double, 2> *px,
                                const std::complex<double> *input,
                                const std::complex<double> *prephase_optional,std::array<IntType, 2> grid_size,
                                HostView<std::complex<double>, 2> padded_grid);

template void spread<double, 3>(NeonufftKernelType kernel_type,
                                const KernelParameters<double> &kernel_param,
                                IntType num_nu, const Point<double, 3> *px,
                                const std::complex<double> *input,
                                const std::complex<double> *prephase_optional,std::array<IntType, 3> grid_size,
                                HostView<std::complex<double>, 3> padded_grid);

#endif

} // namespace neonufft
