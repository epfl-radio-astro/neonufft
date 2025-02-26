#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>

#include "neonufft/config.h"
//---

#include "contrib/es_kernel/util.hpp"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/exceptions.hpp"
#include "neonufft/gpu/device_allocator.hpp"
#include "neonufft/gpu/kernels/interpolation_kernel.hpp"
#include "neonufft/gpu/kernels/upsample_kernel.hpp"
#include "neonufft/gpu/memory/copy.hpp"
#include "neonufft/gpu/memory/device_array.hpp"
#include "neonufft/gpu/plan.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/kernels/interpolation_kernel.hpp"
#include "neonufft/kernels/rescale_loc_kernel.hpp"
#include "neonufft/kernels/upsample_kernel.hpp"
#include "neonufft/memory/array.hpp"
#include "neonufft/memory/view.hpp"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "neonufft/util/fft_grid.hpp"
#include "neonufft/util/point.hpp"
#include "neonufft/util/spread_padding.hpp"

namespace neonufft {
namespace gpu {

template <typename T, std::size_t DIM> class PlanImpl {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM == 1 || DIM == 2 || DIM == 3);

  PlanImpl(Options opt, int sign, IntType num_nu, std::array<const T*, DIM> loc,
           std::array<IntType, DIM> modes, api::StreamType stream,
           std::shared_ptr<Allocator> device_alloc)
      : device_alloc_(std::move(device_alloc)),
        opt_(opt),
        stream_(stream),
        modes_(modes),
        sign_(sign) {
    kernel_param_ = KernelParameters<T>(opt_.tol, opt.upsampfac, opt.kernel_approximation);

    int device_id = 0;
    api::get_device(&device_id);
    api::get_device_properties(&device_prop_, device_id);


    this->set_modes(modes);
    this->set_nu_points(num_nu, loc);
  }

  void transform_type_1(const ComplexType<T>* in, ComplexType<T>* out,
                        std::array<IntType, DIM> out_strides) {
    throw NotImplementedError();
  }

  void transform_type_2(const ComplexType<T>* in, std::array<IntType, DIM> in_strides,
                        ComplexType<T>* out) {
    fft_grid_.padded_view().zero(stream_);

    std::array<ConstDeviceView<T, 1>, DIM> correction_factor_views;
    for (IntType dim = 0; dim < DIM; ++dim) {
      correction_factor_views[dim] = correction_factors_[dim].view();
    }

    ConstDeviceView<ComplexType<T>, DIM> in_view;
    in_view = ConstDeviceView<ComplexType<T>, DIM>(in, modes_, in_strides);

    gpu::upsample<T, DIM>(device_prop_, stream_, opt_.order, in_view, correction_factor_views,
                          fft_grid_.view());



    //---
    HostArray<std::complex<T>, DIM> fft_grid_host(fft_grid_.shape());
    //---

    // HostArray<std::complex<T>, DIM> in_host(in_view.shape());
    // memcopy(in_view, in_host, stream_);
    // std::array<HostArray<T, 1>, DIM> correction_factors_host;
    // std::array<ConstHostView<T, 1>, DIM> correction_factor_views_host;
    // for (std::size_t d = 0; d < DIM; ++d) {
    //   correction_factors_host[d].reset(correction_factors_[d].shape());
    //   correction_factor_views_host[d] = correction_factors_host[d];
    //   memcopy(correction_factors_[d], correction_factors_host[d], stream_);
    // }
    // api::stream_synchronize(stream_);

    // ::neonufft::upsample<T, DIM>(opt_.order, in_host, correction_factor_views_host, fft_grid_host);
    // memcopy(fft_grid_host, fft_grid_.view(), stream_);

    //---


    fft_grid_.transform();

    //---
    // memcopy(fft_grid_.view(), fft_grid_host, stream_);
    // HostArray<Point<T, DIM>, 1> nu_loc_host(nu_loc_.shape());
    // memcopy(nu_loc_, nu_loc_host, stream_);
    // api::stream_synchronize(stream_);

    // HostArray<std::complex<T>, 1> out_host(nu_loc_.shape());

    // ::neonufft::interpolate<T, DIM>(opt_.kernel_type, kernel_param_, fft_grid_host,
    //                                 nu_loc_host.shape(0), nu_loc_host.data(), out_host.data());

    // memcopy(out_host, DeviceView<ComplexType<T>, 1>(out, nu_loc_.size(), 1), stream_);
    // api::stream_synchronize(stream_);
    //---


    gpu::interpolation<T, DIM>(device_prop_, stream_, kernel_param_, nu_loc_, fft_grid_.view(),
                               DeviceView<ComplexType<T>, 1>(out, nu_loc_.size(), 1));
  }

  void set_modes(std::array<IntType, DIM> modes) {
    // required fft grid size
    std::array<IntType, DIM> fft_grid_size;
    for (std::size_t d = 0; d < DIM; ++d) {
      fft_grid_size[d] =
          std::max<std::size_t>(2 * kernel_param_.n_spread, modes[d] * opt_.upsampfac);
    }

    // create new grid if different
    bool new_grid = false;
    for (std::size_t d = 0; d < DIM; ++d) {
      new_grid |= (fft_grid_size[d] != fft_grid_.shape(d));
    }
    if (new_grid) {
      const auto padding = spread_padding(kernel_param_.n_spread);
      typename decltype(fft_grid_)::IndexType pad;
      pad.fill(padding);
      fft_grid_ = FFTGrid<T, DIM>(device_alloc_, stream_, fft_grid_size, sign_, pad);

      // recompute correction factor for kernel windowing
      // we compute the inverse to use multiplication during execution

      std::array<HostArray<T, 1>, DIM> correction_factors_host;
      for (std::size_t d = 0; d < DIM; ++d) {
        auto correction_fact_size = fft_grid_size[d] / 2 + 1;
        correction_factors_[d].reset(correction_fact_size, device_alloc_);
        correction_factors_host[d].reset(correction_fact_size);

        contrib::onedim_fseries_kernel_inverse(fft_grid_size[d], correction_factors_host[d].data(),
                                               kernel_param_.n_spread, kernel_param_.es_halfwidth,
                                               kernel_param_.es_beta, kernel_param_.es_c);
        gpu::memcopy(correction_factors_host[d], correction_factors_[d], stream_);
      }
      // make sure copies are done before correction_factors_host is destroyed
      api::stream_synchronize(stream_);
    }

    modes_ = modes;
  }

  void set_nu_points(IntType num_nu, std::array<const T*, DIM> loc) {
    std::array<HostArray<T, 1>, DIM> loc_host;
    for (std::size_t d = 0; d < DIM; ++d) {
      loc_host[d].reset(num_nu);
      gpu::memcopy(ConstView<T, 1>(loc[d], num_nu, 1), loc_host[d], stream_);
    }
    api::stream_synchronize(stream_);

    HostArray<Point<T, DIM>, 1> nu_loc_host(num_nu);
    if (nu_loc_.shape(0) != num_nu) {
      nu_loc_.reset(num_nu, device_alloc_);
    }

    rescale_loc<T, DIM>(num_nu, loc, nu_loc_host.data());
    gpu::memcopy(nu_loc_host, nu_loc_, stream_);
    api::stream_synchronize(stream_);
  }

private:
  std::shared_ptr<Allocator> device_alloc_;
  Options opt_;
  api::StreamType stream_;
  std::array<IntType, DIM> modes_;
  DeviceArray<Point<T, DIM>, 1> nu_loc_;
  std::vector<PartitionGroup> nu_partition_;
  std::array<DeviceArray<T, 1>, DIM> correction_factors_;
  KernelParameters<T> kernel_param_;
  int sign_;
  FFTGrid<T, DIM> fft_grid_;
  api::DevicePropType device_prop_;
};

}  // namespace gpu
}  // namespace neonufft
