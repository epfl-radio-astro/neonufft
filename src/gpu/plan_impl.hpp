#pragma once

#include "neonufft/config.h"
#include "neonufft/gpu/plan.hpp"

#include "es_kernel/util.hpp"
#include "es_kernel_param.hpp"
#include "gpu/util/runtime_api.hpp"
#include "kernels/rescale_loc_kernel.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "neonufft/exceptions.hpp"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "util/fft_grid.hpp"
#include "util/point.hpp"
#include "util/spread_padding.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>

namespace neonufft {
namespace gpu {

template <typename T, std::size_t DIM> class PlanImpl {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM == 1 || DIM == 2 || DIM == 3);

  PlanImpl(Options opt, int sign, IntType num_nu,
           std::array<const T *, DIM> loc, std::array<IntType, DIM> modes,
           api::StreamType stream)
      : opt_(opt), stream_(stream), modes_(modes), sign_(sign) {

    kernel_param_ =
        KernelParameters<T>(opt_.tol, opt.upsampfac, opt.kernel_approximation);

    this->set_modes(modes);
    this->set_nu_points(num_nu, loc);
  }

  void transform_type_1(const ComplexType<T> *in, ComplexType<T> *out,
                        std::array<IntType, DIM> out_strides) {
    throw NotImplementedError();
  }

  void transform_type_2(const ComplexType<T> *in,
                        std::array<IntType, DIM> in_strides,
                        ComplexType<T> *out) {
    fft_grid_.view().zero();

    std::array<ConstHostView<T, 1>, DIM> correction_factor_views;
    for (IntType dim = 0; dim < DIM; ++dim) {
      correction_factor_views[dim] = correction_factors_[dim].view();
    }

    ConstHostView<ComplexType<T>, DIM> in_view;
    if constexpr (DIM == 1) {
      in_view =
          ConstHostView<ComplexType<T>, DIM>(in, modes_[0], in_strides[0]);
    } else {
      in_view = ConstHostView<ComplexType<T>, DIM>(in, modes_, in_strides);
    }

    // upsample<T, DIM>(opt_.order, in_view, correction_factor_views,
    //                  fft_grid_.view());
    fft_grid_.transform();

    // interpolate<T, DIM>(opt_.kernel_type, kernel_param_, fft_grid_.view(),
    //                     nu_loc_.shape(0), nu_loc_.data(), out);
  }

  void set_modes(std::array<IntType, DIM> modes) {
    // required fft grid size
    std::array<IntType, DIM> fft_grid_size;
    for (std::size_t d = 0; d < DIM; ++d) {
      fft_grid_size[d] = std::max<std::size_t>(2 * kernel_param_.n_spread,
                                               modes[d] * opt_.upsampfac);
    }

    // create new grid if different
    bool new_grid = false;
    for (std::size_t d = 0; d < DIM; ++d) {
      new_grid |= (fft_grid_size[d] != fft_grid_.shape(d));
    }
    if (new_grid) {
      const auto padding = spread_padding(kernel_param_.n_spread);
      if constexpr (DIM == 1) {
        fft_grid_ = FFTGrid<T, DIM>(opt_.num_threads, fft_grid_size[0], sign_, padding);
      } else {
        typename decltype(fft_grid_)::IndexType pad;
        pad.fill(padding);
        fft_grid_ =
            FFTGrid<T, DIM>(opt_.num_threads, fft_grid_size, sign_, pad);
      }

      // recompute correction factor for kernel windowing
      // we compute the inverse to use multiplication during execution

      for (std::size_t d = 0; d < DIM; ++d) {
        auto correction_fact_size = fft_grid_size[d] / 2 + 1;
        correction_factors_[d].reset(correction_fact_size);

        contrib::onedim_fseries_kernel_inverse(
            fft_grid_size[d], correction_factors_[d].data(),
            kernel_param_.n_spread, kernel_param_.es_halfwidth,
            kernel_param_.es_beta, kernel_param_.es_c);
      }
    }

    modes_ = modes;
  }

  void set_nu_points(IntType num_nu, std::array<const T *, DIM> loc) {
    if (nu_loc_.shape(0) != num_nu) {
      nu_loc_ = HostArray<Point<T, DIM>, 1>(num_nu);
    }

    rescale_loc<T, DIM>(num_nu, loc, nu_loc_.data());
  }

private:
  Options opt_;
  api::StreamType stream_;
  std::array<IntType, DIM> modes_;
  HostArray<Point<T,DIM>, 1> nu_loc_;
  std::vector<PartitionGroup> nu_partition_;
  std::array<HostArray<T, 1>, DIM> correction_factors_;
  KernelParameters<T> kernel_param_;
  int sign_;
  FFTGrid<T, DIM> fft_grid_;
};

} // namespace gpu
} // namespace neonufft
