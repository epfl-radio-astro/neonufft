#pragma once

#include "neonufft/config.h"
//--

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>

#include "contrib/es_kernel/util.hpp"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/exceptions.hpp"
#include "neonufft/gpu//util/fft_grid.hpp"
#include "neonufft/gpu/device_allocator.hpp"
#include "neonufft/gpu/kernels/downsample_kernel.hpp"
#include "neonufft/gpu/kernels/interpolation_kernel.hpp"
#include "neonufft/gpu/kernels/rescale_loc_kernel.hpp"
#include "neonufft/gpu/kernels/spreading_kernel.hpp"
#include "neonufft/gpu/kernels/upsample_kernel.hpp"
#include "neonufft/gpu/kernels/min_max_kernel.hpp"
#include "neonufft/gpu/memory/copy.hpp"
#include "neonufft/gpu/memory/device_array.hpp"
#include "neonufft/gpu/plan.hpp"
#include "neonufft/gpu/util/partition_group.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/kernels/downsample_kernel.hpp"
#include "neonufft/kernels/interpolation_kernel.hpp"
#include "neonufft/kernels/rescale_loc_kernel.hpp"
#include "neonufft/kernels/spreading_kernel.hpp"
#include "neonufft/kernels/upsample_kernel.hpp"
#include "neonufft/memory/array.hpp"
#include "neonufft/memory/view.hpp"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "neonufft/util/grid_info_t3.hpp"
#include "neonufft/util/point.hpp"

namespace neonufft {
namespace gpu {

template <typename T, std::size_t DIM>
class PlanT3Impl {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM == 1 || DIM == 2 || DIM == 3);

  static std::uint64_t grid_memory_size(const Options& opt, std::array<T, DIM> input_min,
                                        std::array<T, DIM> input_max, std::array<T, DIM> output_min,
                                        std::array<T, DIM> output_max) {
    KernelParameters<T> kernel_param(opt.tol, opt.upsampfac, opt.kernel_approximation);
    GridInfoT3<T, DIM> grid_info(kernel_param.n_spread, opt.upsampfac, opt.recenter_threshold,
                                 input_min, input_max, output_min, output_max);

    const std::uint64_t total_spread_grid_size =
        std::reduce(grid_info.spread_grid_size.begin(), grid_info.spread_grid_size.end(), 1,
                    std::multiplies<IntType>());

    const std::uint64_t total_fft_grid_size =
        std::reduce(grid_info.fft_grid_size.begin(), grid_info.fft_grid_size.end(), 1,
                    std::multiplies<IntType>());

    return sizeof(ComplexType<T>) * (total_spread_grid_size + total_fft_grid_size);
  }

  PlanT3Impl(Options opt, int sign, IntType num_in, std::array<const T*, DIM> input_points,
             IntType num_out, std::array<const T*, DIM> output_points, api::StreamType stream,
             std::shared_ptr<Allocator> device_alloc)
      : opt_(opt),
        stream_(stream),
        device_alloc_(std::move(device_alloc)),
        sign_(sign),
        kernel_param_(opt_.tol, opt_.upsampfac, opt.kernel_approximation) {
    int device_id = 0;
    api::get_device(&device_id);
    api::get_device_properties(&device_prop_, device_id);

    DeviceArray<T, 1> input_buffer(min_max_worksize<T>(num_in), device_alloc_);
    DeviceArray<T, 1> output_buffer(min_max_worksize<T>(num_out), device_alloc_);
    DeviceArray<T, 1> min_max_buffer(4 * DIM, device_alloc_);

    for (IntType d = 0; d < DIM; ++d) {
      min_max<T>(ConstDeviceView<T, 1>(input_points[d], num_in), min_max_buffer.data() + 2 * d,
                 min_max_buffer.data() + 2 * d + 1, input_buffer.data(), stream_);
      min_max<T>(ConstDeviceView<T, 1>(output_points[d], num_out),
                 min_max_buffer.data() + 2 * DIM + 2 * d,
                 min_max_buffer.data() + 2 * DIM + 2 * d + 1, output_buffer.data(), stream_);
    }

    HostArray<T, 1> min_max_host_buffer(min_max_buffer.shape());
    memcopy(min_max_buffer, min_max_host_buffer, stream_);
    api::stream_synchronize(stream_;);

    std::array<T, DIM> input_min, input_max, output_min, output_max;

    for (IntType d = 0; d < DIM; ++d) {
      input_min[d] = min_max_host_buffer[2 * d];
      input_max[d] = min_max_host_buffer[2 * d + 1];
      output_min[d] = min_max_host_buffer[2 * d + 2 * DIM];
      output_max[d] = min_max_host_buffer[2 * d + 2 * DIM];
    }

    grid_info_ = GridInfo(kernel_param_.n_spread, opt_.upsampfac, opt_.recenter_threshold,
                          input_min, input_max, output_min, output_max);

    this->init(input_min, input_max, output_min, output_max);
    this->set_input_points(num_in, input_points);
    this->set_output_points(num_out, output_points);
  }

  PlanT3Impl(Options opt, int sign, std::array<T, DIM> input_min, std::array<T, DIM> input_max,
             std::array<T, DIM> output_min, std::array<T, DIM> output_max, api::StreamType stream,
             std::shared_ptr<Allocator> device_alloc)
      : opt_(opt),
        stream_(stream),
        device_alloc_(std::move(device_alloc)),
        sign_(sign),
        kernel_param_(opt_.tol, opt_.upsampfac, opt_.kernel_approximation),
        grid_info_(kernel_param_.n_spread, opt_.upsampfac, opt_.recenter_threshold, input_min,
                   input_max, output_min, output_max) {
    int device_id = 0;
    api::get_device(&device_id);
    api::get_device_properties(&device_prop_, device_id);

    this->init(input_min, input_max, output_min, output_max);
  }

  void set_input_points(IntType num_in, std::array<const T*, DIM> input_points) {
    // compute prephase
    if (grid_info_.output_offsets[0] != 0 ||
        grid_info_.output_offsets[std::min<IntType>(1, DIM - 1)] != 0 ||
        grid_info_.output_offsets[std::min<IntType>(2, DIM - 1)] != 0) {
      prephase_.reset(num_in);
      compute_prephase<T, DIM>(sign_, num_in, input_points, grid_info_.output_offsets,
                               prephase_.data());
    }

    // first translate rescale for grid spacing adjustment, then scale points to
    // [0, 1]
    std::array<T, DIM> input_scaling_factors;
    for (IntType d = 0; d < DIM; ++d) {
      input_scaling_factors[d] = 1 / gam_[d];
    }
    if (rescaled_input_points_.shape(0) != num_in) {
      rescaled_input_points_.reset(num_in);
    }

    // TODO partitioning
  }

  void set_output_points(IntType num_out, std::array<const T*, DIM> output_points) {
    std::array<T, DIM> output_scaling_factors;
    for (IntType d = 0; d < DIM; ++d) {
      output_scaling_factors[d] = gam_[d] * spread_grid_spacing_[d];
    }
    if (rescaled_output_points_.shape(0) != num_out) {
      rescaled_output_points_.reset(num_out);
    }

    const auto output_partition = rescale_loc_partition_t3<T, DIM>(
        DIM - 1, 4 * th_pool_.num_threads(), num_out, grid_info_.output_offsets,
        output_scaling_factors, output_points, rescaled_output_points_.data());


    // compute postphase with correction factors
    {
      HostArray<T, 1> phi_hat(num_out);
      postphase_.reset(num_out);
      nuft_real<T, DIM>(opt_.kernel_type, kernel_param_, num_out, output_points,
                        grid_info_.output_offsets, output_scaling_factors, phi_hat.data());

      if (grid_info_.input_offsets[0] != 0 ||
          grid_info_.input_offsets[std::min<IntType>(1, DIM - 1)] != 0 ||
          grid_info_.input_offsets[std::min<IntType>(2, DIM - 1)] != 0) {
        compute_postphase<T, DIM>(sign_, num_out, phi_hat.data(), output_points,
                                  grid_info_.input_offsets, grid_info_.output_offsets,
                                  postphase_.data());
      } else {
        // just store inverse of phi_hat
        const T* cf_ptr = phi_hat.data();
        auto pp_ptr = postphase_.data();
        for (IntType i = 0; i < num_out; ++i) {
          pp_ptr[i] = ComplexType<T>{1 / cf_ptr[i], 0};
        }
      }
    }
  }

  void transform(ComplexType<T>* out) {
    fft_grid_.view().zero(stream_);

    // TODO: folding might not be needed. Do we ever have points on the actual
    // edges? check how points are rescaled. Unit tests show values in padding
    // is always 0
    // fold_padding<T, DIM>(kernel_param_.n_spread, spread_grid_);

    std::array<ConstHostView<T, 1>, DIM> correction_factor_views;
    for (IntType dim = 0; dim < DIM; ++dim) {
      correction_factor_views[dim] = correction_factors_[dim].view();
    }

    const auto padding = spread_padding(kernel_param_.n_spread);
    if constexpr (DIM == 1) {
      upsample<T, DIM>(NEONUFFT_MODE_ORDER_CMCL,
                       spread_grid_.sub_view(padding, grid_info_.spread_grid_size[0]),
                       correction_factor_views, fft_grid_.view());
    } else if constexpr (DIM == 2) {
      upsample<T, DIM>(NEONUFFT_MODE_ORDER_CMCL,
                       spread_grid_.sub_view({padding, padding}, {grid_info_.spread_grid_size[0],
                                                                  grid_info_.spread_grid_size[1]}),
                       correction_factor_views, fft_grid_.view());
    } else {
      upsample<T, DIM>(
          NEONUFFT_MODE_ORDER_CMCL,
          spread_grid_.sub_view({padding, padding, padding},
                                {grid_info_.spread_grid_size[0], grid_info_.spread_grid_size[1],
                                 grid_info_.spread_grid_size[2]}),
          correction_factor_views, fft_grid_.view());
    }

    fft_grid_.transform();

    interpolate<T, DIM>(opt_.kernel_type, kernel_param_, fft_grid_.view(),
                        rescaled_output_points_.shape(0), rescaled_output_points_.data(), out);

    // TODO: move to kernel. NOTE: integration into interpolate functions
    // results in large performance hit with clang (18)
    for (IntType i = 0; i < postphase_.shape(0); ++i) {
      out[i] *= postphase_[i];
    }

    // TODO: reset function
    //  Reset spread grid
    spread_grid_.zero(stream_);
  }

  void add_input(const ComplexType<T>* in) {
    ConstDeviceView<ComplexType<T>, 1> in_view(in, rescaled_input_points_.shape(0), 1);
    gpu::spread<T, DIM>(device_prop_, stream_, kernel_param_, partition_, rescaled_input_points_,
                        in_view, prephase_, fft_grid_.view());
  }

private:
  void init(std::array<T, DIM> input_min, std::array<T, DIM> input_max,
            std::array<T, DIM> output_min, std::array<T, DIM> output_max) {
    for (IntType d = 0; d < DIM; ++d) {
      if (input_min[d] > input_max[d]) {
        throw InputError("Invalid input bounds.");
      }
      if (output_min[d] > output_max[d]) {
        throw InputError("Invalid output bounds.");
      }
    }

    for (IntType d = 0; d < DIM; ++d) {
      contrib::set_nhg_type3<T>(grid_info_.output_half_extent[d], grid_info_.input_half_extent[d],
                                opt_.upsampfac, grid_info_.spread_grid_size[d],
                                &spread_grid_spacing_[d], &gam_[d]);
    }

    // reshape spread grid
    {
      bool new_grid = false;
      for (std::size_t d = 0; d < DIM; ++d) {
        new_grid |= (spread_grid_.shape(d) != grid_info_.padded_spread_grid_size[d]);
      }
      if (new_grid) {
        spread_grid_.reset(grid_info_.padded_spread_grid_size);
      }
    }

    // zero spread grid to prepare accumulation of input data
    spread_grid_.zero(stream_);

    // reshape fft grid
    {
      // create new grid if different
      bool new_grid = false;
      for (std::size_t d = 0; d < DIM; ++d) {
        new_grid |= (grid_info_.fft_grid_size[d] != fft_grid_.shape(d));
      }
      if (new_grid) {
        fft_grid_ = FFTGrid<T, DIM>(device_alloc_, grid_info_.fft_grid_size, sign_);
      }
    }

    // recompute correction factor for kernel windowing
    // we compute the inverse to use multiplication during execution
    std::array<HostArray<T, 1>, DIM> correction_factors_host;
    for (std::size_t d = 0; d < DIM; ++d) {
      auto correction_fact_size = fft_grid_.shape(d) / 2 + 1;
      correction_factors_host[d].reset(correction_fact_size);
      if (correction_factors_[d].shape(0) != correction_fact_size) {
        correction_factors_[d].reset(correction_fact_size);
      }

      contrib::onedim_fseries_kernel_inverse(fft_grid_.shape(d), correction_factors_host[d].data(),
                                             kernel_param_.n_spread, kernel_param_.es_halfwidth,
                                             kernel_param_.es_beta, kernel_param_.es_c);
      memcopy(correction_factors_host[d], correction_factors_[d], stream_);
    }
    api::stream_synchronize(stream_);
  }

  Options opt_;
  api::DevicePropType device_prop_;
  std::shared_ptr<Allocator> device_alloc_;
  api::StreamType stream_;
  DeviceArray<Point<T, DIM>, 1> rescaled_input_points_;
  DeviceArray<Point<T, DIM>, 1> rescaled_output_points_;
  DeviceArray<ComplexType<T>, 1> prephase_;
  DeviceArray<ComplexType<T>, 1> postphase_;
  std::array<DeviceArray<T, 1>, DIM> correction_factors_;
  int sign_;
  KernelParameters<T> kernel_param_;
  FFTGrid<T, DIM> fft_grid_;
  GridInfoT3<T, DIM> grid_info_;
  DeviceArray<ComplexType<T>, DIM> spread_grid_;
  std::array<T, DIM> gam_;
  std::array<T, DIM> spread_grid_spacing_;
  DeviceArray<PartitionGroup, DIM> partition_;
};

// TODO remove
template class PlanT3Impl<float, 1>;
template class PlanT3Impl<float, 2>;
template class PlanT3Impl<float, 3>;
template class PlanT3Impl<double, 1>;
template class PlanT3Impl<double, 2>;
template class PlanT3Impl<double, 3>;

}  // namespace gpu
}  // namespace neonufft
