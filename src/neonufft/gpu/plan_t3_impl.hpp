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
#include "neonufft/gpu/device_allocator.hpp"
#include "neonufft/gpu/kernels/downsample_kernel.hpp"
#include "neonufft/gpu/kernels/interpolation_kernel.hpp"
#include "neonufft/gpu/kernels/min_max_kernel.hpp"
#include "neonufft/gpu/kernels/postphase_kernel.hpp"
#include "neonufft/gpu/kernels/prephase_kernel.hpp"
#include "neonufft/gpu/kernels/rescale_loc_kernel.hpp"
#include "neonufft/gpu/kernels/spreading_kernel.hpp"
#include "neonufft/gpu/kernels/upsample_kernel.hpp"
#include "neonufft/gpu/kernels/fseries_kernel.hpp"
#include "neonufft/gpu/memory/copy.hpp"
#include "neonufft/gpu/memory/device_array.hpp"
#include "neonufft/gpu/plan.hpp"
#include "neonufft/gpu/util/fft_grid.hpp"
#include "neonufft/gpu/util/partition_group.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/kernels/compute_postphase_kernel.hpp"
#include "neonufft/kernels/compute_prephase_kernel.hpp"
#include "neonufft/kernels/downsample_kernel.hpp"
#include "neonufft/kernels/interpolation_kernel.hpp"
#include "neonufft/kernels/nuft_real_kernel.hpp"
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

    std::uint64_t total_spread_grid_size = 1;
    std::uint64_t total_fft_grid_size = 1;

    for (IntType d = 0; d < DIM; ++d) {
      total_spread_grid_size *= grid_info.spread_grid_size[d];
      total_fft_grid_size *= grid_info.fft_grid_size[d];
    }

    // We multiply the fft grid size with 2, because cufft will require a workspace area of equal
    // size for in-place transforms
    return sizeof(ComplexType<T>) * (total_spread_grid_size + 2 * total_fft_grid_size);
  }

  PlanT3Impl(IntType batch_size, Options opt, int sign, IntType num_in,
             std::array<const T*, DIM> input_points, IntType num_out,
             std::array<const T*, DIM> output_points, api::StreamType stream,
             std::shared_ptr<Allocator> device_alloc)
      : batch_size_(batch_size),
        opt_(opt),
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
      min_max<T>(ConstDeviceView<T, 1>(input_points[d], num_in, 1), min_max_buffer.data() + 2 * d,
                 min_max_buffer.data() + 2 * d + 1, input_buffer, stream_);
      min_max<T>(ConstDeviceView<T, 1>(output_points[d], num_out, 1),
                 min_max_buffer.data() + 2 * DIM + 2 * d,
                 min_max_buffer.data() + 2 * DIM + 2 * d + 1, output_buffer, stream_);
    }

    HostArray<T, 1> min_max_host_buffer(min_max_buffer.shape());
    memcopy(min_max_buffer, min_max_host_buffer, stream_);
    api::stream_synchronize(stream_);

    std::array<T, DIM> input_min, input_max, output_min, output_max;

    for (IntType d = 0; d < DIM; ++d) {
      input_min[d] = min_max_host_buffer[2 * d];
      input_max[d] = min_max_host_buffer[2 * d + 1];
      output_min[d] = min_max_host_buffer[2 * d + 2 * DIM];
      output_max[d] = min_max_host_buffer[2 * d + 2 * DIM + 1];
    }

    grid_info_ = GridInfoT3<T, DIM>(kernel_param_.n_spread, opt_.upsampfac, opt_.recenter_threshold,
                                    input_min, input_max, output_min, output_max);

    this->init(input_min, input_max, output_min, output_max);
    this->set_input_points(num_in, input_points);
    this->set_output_points(num_out, output_points);
  }

  PlanT3Impl(IntType batch_size, Options opt, int sign, std::array<T, DIM> input_min,
             std::array<T, DIM> input_max, std::array<T, DIM> output_min,
             std::array<T, DIM> output_max, api::StreamType stream,
             std::shared_ptr<Allocator> device_alloc)
      : batch_size_(batch_size),
        opt_(opt),
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
    StackArray<ConstDeviceView<T, 1>, DIM> input_point_views;
    for (IntType d = 0; d < DIM; ++d) {
      input_point_views[d] = ConstDeviceView<T, 1>(input_points[d], num_in, 1);
    }

    // compute prephase
    if (grid_info_.output_offsets[0] != 0 ||
        grid_info_.output_offsets[std::min<IntType>(1, DIM - 1)] != 0 ||
        grid_info_.output_offsets[std::min<IntType>(2, DIM - 1)] != 0) {
      prephase_.reset(num_in, device_alloc_);


      gpu::compute_prephase<T, DIM>(device_prop_, stream_, sign_, input_point_views,
                                    grid_info_.output_offsets, prephase_);
    }


    // first translate rescale for grid spacing adjustment, then scale points to
    // [0, 1]
    StackArray<T, DIM> input_scaling_factors;
    for (IntType d = 0; d < DIM; ++d) {
      input_scaling_factors[d] = 1 / gam_[d];
    }
    if (rescaled_input_points_.shape(0) != num_in) {
      rescaled_input_points_.reset(num_in, device_alloc_);
    }

    // StackArray<ConstDeviceView<T, 1>, DIM> loc_views;
    // for(IntType d = 0; d <DIM; ++d) {
    //   loc_views[d] = ConstDeviceView<T, 1>(input_points[d], num_in, 1);
    // }
    // rescale_and_permut_t3<T, DIM>(device_prop_, stream_, loc_views, spread_grid_.shape(),
    //                               grid_info_.input_offsets, input_scaling_factors, input_partition_,
    //                               rescaled_input_points_);

    rescale_t3<T, DIM>(device_prop_, stream_, input_point_views, grid_info_.spread_grid_size,
                       grid_info_.input_offsets, input_scaling_factors, rescaled_input_points_);
  }

  void set_output_points(IntType num_out, std::array<const T*, DIM> output_points) {
    std::array<T, DIM> output_scaling_factors;
    for (IntType d = 0; d < DIM; ++d) {
      output_scaling_factors[d] = gam_[d] * spread_grid_spacing_[d];
    }
    if (rescaled_output_points_.shape(0) != num_out) {
      rescaled_output_points_.reset(num_out, device_alloc_);
    }

    StackArray<ConstDeviceView<T, 1>, DIM> loc_views;
    for(IntType d = 0; d <DIM; ++d) {
      loc_views[d] = ConstDeviceView<T, 1>(output_points[d], num_out, 1);
    }


    // TODO: output partition not needed. Still sort for better caching?
    // IndexArray<DIM> output_partition_shape;
    // // output_partition_shape.fill(1);
    // output_partition_shape = input_partition_.shape();

    // DeviceArray<PartitionGroup, DIM> output_partition(output_partition_shape, device_alloc_);
    // rescale_and_permut_t3<T, DIM>(device_prop_, stream_, loc_views, fft_grid_.shape(),
    //                               grid_info_.output_offsets, output_scaling_factors,
    //                               output_partition, rescaled_output_points_);

    rescale_t3<T, DIM>(device_prop_, stream_, loc_views, grid_info_.spread_grid_size,
                       grid_info_.output_offsets, output_scaling_factors, rescaled_output_points_);

    // compute postphase with correction factors
    postphase_.reset(num_out, device_alloc_);

    gpu::postphase<T, DIM>(device_prop_, stream_, kernel_param_, sign_, loc_views,
                           grid_info_.input_offsets, grid_info_.output_offsets,
                           output_scaling_factors, postphase_);
  }

  void transform(ComplexType<T>* out, IntType bdist) {
    if (bdist <= 0) bdist = rescaled_output_points_.shape(0);


    std::array<ConstDeviceView<T, 1>, DIM> correction_factor_views;
    for (IntType dim = 0; dim < DIM; ++dim) {
      correction_factor_views[dim] = correction_factors_[dim].view();
    }

    for (IntType idx_batch = 0; idx_batch < batch_size_; ++idx_batch) {
      fft_grid_.view().zero(stream_);
      gpu::upsample<T, DIM>(device_prop_, stream_, NEONUFFT_MODE_ORDER_CMCL,
                            spread_grid_.slice_view(idx_batch), correction_factor_views,
                            fft_grid_.view());

      fft_grid_.transform();

      gpu::interpolation<T, DIM>(device_prop_, stream_, kernel_param_, rescaled_output_points_,
                                 fft_grid_.view(), postphase_,
                                 DeviceView<ComplexType<T>, 1>(out + idx_batch * bdist,
                                                               rescaled_output_points_.size(), 1));
    }

    // TODO: reset function
    //  Reset spread grid
    spread_grid_.view().zero(stream_);
  }

  void add_input(const ComplexType<T>* in, IntType bdist) {
    if (bdist <= 0) bdist = rescaled_input_points_.shape(0);

    for (IntType idx_batch = 0; idx_batch < batch_size_; ++idx_batch) {
      ConstDeviceView<ComplexType<T>, 1> in_view(in + idx_batch * bdist,
                                                 rescaled_input_points_.shape(0), 1);
      gpu::spread<T, DIM>(device_prop_, stream_, kernel_param_, input_partition_,
                          rescaled_input_points_, in_view, prephase_,
                          spread_grid_.slice_view(idx_batch));
    }
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
    IndexArray<DIM + 1> batched_shape;
    batched_shape[DIM] = batch_size_;
    for (IntType d = 0; d < DIM; ++d) {
      // no padding on GPU
      batched_shape[d] = grid_info_.spread_grid_size[d];
    }
    spread_grid_.reset(batched_shape, device_alloc_);

    // zero spread grid to prepare accumulation of input data
    spread_grid_.zero(stream_);

    // reshape fft grid
    fft_grid_ = FFTGrid<T, DIM>(device_alloc_, stream_, grid_info_.fft_grid_size, sign_);

    // recompute correction factor for kernel windowing
    // we compute the inverse to use multiplication during execution
    for (std::size_t d = 0; d < DIM; ++d) {
      const auto correction_fact_size = fft_grid_.shape(d) / 2 + 1;
      correction_factors_[d].reset(correction_fact_size, device_alloc_);

      gpu::fseries_inverse<T>(device_prop_, stream_, kernel_param_, fft_grid_.shape(d),
                              correction_factors_[d]);
    }

    // resize partition grid
    typename decltype(input_partition_)::IndexType part_grid_size;
    for (std::size_t d = 0; d < DIM; ++d) {
      part_grid_size[d] =
          (spread_grid_.shape(d) + gpu::PartitionGroup::width - 1) / PartitionGroup::width;
    }
    input_partition_.reset(part_grid_size, device_alloc_);

    api::stream_synchronize(stream_);
  }

  IntType batch_size_;
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
  DeviceArray<ComplexType<T>, DIM + 1> spread_grid_;
  std::array<T, DIM> gam_;
  std::array<T, DIM> spread_grid_spacing_;
  DeviceArray<PartitionGroup, DIM> input_partition_;
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
