#pragma once

#include "neonufft/config.h"

#include "es_kernel/util.hpp"
#include "es_kernel_param.hpp"
#include "kernels/compute_postphase_kernel.hpp"
#include "kernels/compute_prephase_kernel.hpp"
#include "kernels/fold_padding.hpp"
#include "kernels/interpolation_kernel.hpp"
#include "kernels/min_max_kernel.hpp"
#include "kernels/nuft_real_kernel.hpp"
#include "kernels/rescale_loc_kernel.hpp"
#include "kernels/spreading_kernel.hpp"
#include "kernels/upsample_kernel.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"
#include "neonufft/exceptions.hpp"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "threading//thread_pool.hpp"
#include "util/fft_grid.hpp"
#include "util/point.hpp"
#include "util/spread_padding.hpp"
#include "util/zorder.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>

namespace neonufft {

template <typename T, std::size_t DIM> class PlanT3Impl {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM == 1 || DIM == 2 || DIM == 3);

  static std::uint64_t grid_memory_size(const Options &opt,
                                        std::array<T, DIM> input_min,
                                        std::array<T, DIM> input_max,
                                        std::array<T, DIM> output_min,
                                        std::array<T, DIM> output_max) {
    KernelParameters<T> kernel_param(opt.tol, opt.upsampfac,
                                     opt.kernel_approximation);
    GridInfo grid_info(kernel_param.n_spread, opt.upsampfac,
                       opt.recenter_threshold, input_min, input_max, output_min,
                       output_max);

    const std::uint64_t total_spread_grid_size = std::reduce(
        grid_info.padded_spread_grid_size.begin(),
        grid_info.padded_spread_grid_size.end(), 1, std::multiplies<IntType>());

    const std::uint64_t total_fft_grid_size = std::reduce(
        grid_info.fft_grid_size.begin(), grid_info.fft_grid_size.end(), 1,
        std::multiplies<IntType>());

    return sizeof(std::complex<T>) *
           (total_spread_grid_size + total_fft_grid_size);
  }

  PlanT3Impl(Options opt, int sign, IntType num_in,
             std::array<const T *, DIM> input_points, IntType num_out,
             std::array<const T *, DIM> output_points)
      : opt_(opt), th_pool_(opt_.num_threads), sign_(sign),
        kernel_param_(opt_.tol, opt_.upsampfac, opt.kernel_approximation) {

    std::array<T, DIM> input_min, input_max, output_min, output_max;

    for (IntType d = 0; d < DIM; ++d) {
      const auto in_mm = min_max(num_in, input_points[d]);
      input_min[d] = in_mm.first;
      input_max[d] = in_mm.second;
      const auto out_mm = min_max(num_out, output_points[d]);
      output_min[d] = out_mm.first;
      output_max[d] = out_mm.second;
    }

    grid_info_ = GridInfo(kernel_param_.n_spread, opt_.upsampfac,
                          opt_.recenter_threshold, input_min, input_max,
                          output_min, output_max);

    this->init(input_min, input_max, output_min, output_max);
    this->set_input_points(num_in, input_points);
    this->set_output_points(num_out, output_points);

    // TODO: set points
    // this->set_points(num_in, input_points, num_out, output_points);
  }

  PlanT3Impl(Options opt, int sign, std::array<T, DIM> input_min,
             std::array<T, DIM> input_max, std::array<T, DIM> output_min,
             std::array<T, DIM> output_max)
      : opt_(opt), th_pool_(opt_.num_threads), sign_(sign),
        kernel_param_(opt_.tol, opt_.upsampfac, opt_.kernel_approximation),
        grid_info_(kernel_param_.n_spread, opt_.upsampfac,
                   opt_.recenter_threshold, input_min, input_max, output_min,
                   output_max) {
    this->init(input_min, input_max, output_min, output_max);
  }

  void set_input_points(IntType num_in, std::array<const T *, DIM> input_points) {
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

    {
      // The spreading is done in two steps, processing every even group and
      // every odd group. To allow for load balancing, use 3 partitions per
      // thread as target at each step.
      IntType num_input_partitions = 6 *th_pool_.num_threads();

      // spreading requires a mimimum width of n_spread
      const IntType partition_width = kernel_param_.n_spread + 4;

      // find suitable dimension to split. Select outer most dimension that fits
      // for better cache usage.
      IntType input_partition_dim = DIM - 1;
      if constexpr (DIM > 1) {
        if (grid_info_.spread_grid_size[DIM - 1] <
                num_input_partitions * partition_width &&
            grid_info_.spread_grid_size[DIM - 2] >=
                num_input_partitions * partition_width) {
          input_partition_dim = DIM - 2;
        }
      }

      // the width of a partition must be at least n_spread to avoid race
      // condition in spreading kernel
      num_input_partitions = std::min(
          num_input_partitions,
          grid_info_.spread_grid_size[input_partition_dim] / partition_width);

      num_input_partitions = std::max<IntType>(1, num_input_partitions);

      input_partition_ = rescale_loc_partition_t3<T, DIM>(
          input_partition_dim, num_input_partitions, num_in, grid_info_.input_offsets,
          input_scaling_factors, input_points, rescaled_input_points_.data());

    }


    // z order sorting within each group
    if (opt_.sort_input) {
      th_pool_.parallel_for(
          {0, IntType(input_partition_.size())}, 1,
          [&](IntType, BlockRange range) {
            for (IntType idx_group = range.begin; idx_group < range.end;
                 ++idx_group) {
              const auto &g = input_partition_[idx_group];

              std::sort(rescaled_input_points_.data() + g.begin,
                        rescaled_input_points_.data() + g.begin + g.size,
                        [](const auto &p1, const auto &p2) {
                          return zorder::less_0_1<DIM>(p1.coord, p2.coord);
                        });
            }
          });
    }
  }

  void set_output_points(IntType num_out,
                               std::array<const T *, DIM> output_points) {
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

    if (opt_.sort_output) {
      th_pool_.parallel_for(
          {0, IntType(output_partition.size())}, 1,
          [&](IntType, BlockRange range) {
            for (IntType idx_group = range.begin; idx_group < range.end;
                 ++idx_group) {
              const auto &g = output_partition[idx_group];

              std::sort(rescaled_output_points_.data() + g.begin,
                        rescaled_output_points_.data() + g.begin + g.size,
                        [](const auto &p1, const auto &p2) {
                          return zorder::less_0_1<DIM>(p1.coord, p2.coord);
                        });
            }
          });
    }

    // compute postphase with correction factors
    {
      HostArray<T, 1> phi_hat(num_out);
      postphase_.reset(num_out);
      nuft_real<T, DIM>(opt_.kernel_type, kernel_param_, num_out, output_points,
                        grid_info_.output_offsets, output_scaling_factors,
                        phi_hat.data());

      if (grid_info_.input_offsets[0] != 0 ||
          grid_info_.input_offsets[std::min<IntType>(1, DIM - 1)] != 0 ||
          grid_info_.input_offsets[std::min<IntType>(2, DIM - 1)] != 0) {
        compute_postphase<T, DIM>(sign_, num_out, phi_hat.data(), output_points,
                                  grid_info_.input_offsets, grid_info_.output_offsets,
                                  postphase_.data());
      } else {
        // just store inverse of phi_hat
        const T *cf_ptr = phi_hat.data();
        auto pp_ptr = postphase_.data();
        for (IntType i = 0; i < num_out; ++i) {
          pp_ptr[i] = std::complex<T>{1 / cf_ptr[i], 0};
        }
      }
    }
  }

  void transform(std::complex<T> *out) {
    fft_grid_.padded_view().zero();

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
      upsample<T, DIM>(
          NEONUFFT_MODE_ORDER_CMCL,
          spread_grid_.sub_view(padding,
                                grid_info_.spread_grid_size[0]),
          correction_factor_views, fft_grid_.view());
    } else if constexpr (DIM == 2) {
      upsample<T, DIM>(
          NEONUFFT_MODE_ORDER_CMCL,
          spread_grid_.sub_view({padding, padding},
                                {grid_info_.spread_grid_size[0], grid_info_.spread_grid_size[1]}),
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

    if (th_pool_.num_threads() > 1) {
      th_pool_.parallel_for(
          {0, rescaled_output_points_.size()}, [&](IntType, BlockRange range) {
            auto sub_view = rescaled_output_points_.sub_view(
                range.begin, range.end - range.begin);
            interpolate<T, DIM>(opt_.kernel_type, kernel_param_,
                                fft_grid_.view(), sub_view.shape(0),
                                sub_view.data(), out);
          });
    } else {

      interpolate<T, DIM>(opt_.kernel_type, kernel_param_, fft_grid_.view(),
                          rescaled_output_points_.shape(0),
                          rescaled_output_points_.data(), out);
    }
    // TODO: move to kernel. NOTE: integration into interpolate functions
    // results in large performance hit with clang (18)
    for (IntType i = 0; i < postphase_.shape(0); ++i) {
      out[i] *= postphase_[i];
    }

    // Reset spread grid
    spread_grid_.zero();
  }

  void add_input(const std::complex<T> *in) {
    if (th_pool_.num_threads() > 1) {
      // spread even groups
      th_pool_.parallel_for(
          {0, IntType(input_partition_.size())}, 2,
          [&](IntType, BlockRange range) {
            for (IntType idx_group = range.begin; idx_group < range.end;
                 ++idx_group) {
              const auto &g = input_partition_[idx_group];
              if (idx_group % 2 == 0 && g.size) {
                auto sub_view = rescaled_input_points_.sub_view(g.begin, g.size);
                spread<T, DIM>(opt_.kernel_type, kernel_param_,
                               sub_view.shape(0), sub_view.data(), in,
                               prephase_.size() ? prephase_.data() : nullptr,
                               grid_info_.spread_grid_size, spread_grid_);
              }
            }
          });

      // spread odd groups
      th_pool_.parallel_for(
          {0, IntType(input_partition_.size())}, 2,
          [&](IntType, BlockRange range) {
            for (IntType idx_group = range.begin; idx_group < range.end;
                 ++idx_group) {
              const auto &g = input_partition_[idx_group];
              if (idx_group % 2 == 1 && g.size) {
                auto sub_view = rescaled_input_points_.sub_view(g.begin, g.size);
                spread<T, DIM>(opt_.kernel_type, kernel_param_,
                               sub_view.shape(0), sub_view.data(), in,
                               prephase_.size() ? prephase_.data() : nullptr,
                               grid_info_.spread_grid_size, spread_grid_);
              }
            }
          });
    } else {
      spread<T, DIM>(opt_.kernel_type, kernel_param_,
                     rescaled_input_points_.shape(0), rescaled_input_points_.data(),
                     in, prephase_.size() ? prephase_.data() : nullptr,
                     grid_info_.spread_grid_size, spread_grid_);
    }
  }

private:
  struct GridInfo {
    GridInfo() = default;

    GridInfo(IntType n_spread, double upsampfac, T recenter_threshold,
             std::array<T, DIM> input_min_i, std::array<T, DIM> input_max_i,
             std::array<T, DIM> output_min_i, std::array<T, DIM> output_max_i)
        : input_min(input_min_i), input_max(input_max_i),
          output_min(output_min_i), output_max(output_max_i) {

      for (IntType d = 0; d < DIM; ++d) {
        input_offsets[d] = (input_max[d] + input_min[d]) / 2;
        output_offsets[d] = (output_max[d] + output_min[d]) / 2;

        input_half_extent[d] = std::max(std::abs(input_max[d] - input_offsets[d]),
                                     std::abs(input_min[d] - input_offsets[d]));

        output_half_extent[d] =
            std::max(std::abs(output_max[d] - output_offsets[d]),
                     std::abs(output_min[d] - output_offsets[d]));

        if (std::abs(input_offsets[d]) <
            recenter_threshold * input_half_extent[d]) {
          input_offsets[d] = 0;
          input_half_extent[d] =
              std::max(std::abs(input_max[d]), std::abs(input_min[d]));
        }
        if (std::abs(output_offsets[d]) <
            recenter_threshold * output_half_extent[d]) {
          output_offsets[d] = 0;
          output_half_extent[d] =
              std::max(std::abs(output_max[d]), std::abs(output_min[d]));
        }

        spread_grid_size[d] = contrib::t3_grid_size<T>(
            output_half_extent[d], input_half_extent[d], upsampfac, n_spread);

        padded_spread_grid_size[d] =
            spread_grid_size[d] + 2 * spread_padding(n_spread);

        fft_grid_size[d] = contrib::next235even(std::max<std::size_t>(
            2 * n_spread, spread_grid_size[d] * upsampfac));
      }
    }

    std::array<T, DIM> input_min = {0};
    std::array<T, DIM> input_max = {0};
    std::array<T, DIM> output_min = {0};
    std::array<T, DIM> output_max = {0};
    std::array<IntType, DIM> spread_grid_size = {0};
    std::array<IntType, DIM> padded_spread_grid_size = {0};
    std::array<IntType, DIM> fft_grid_size = {0};
    std::array<T, DIM> input_offsets = {0};
    std::array<T, DIM> output_offsets = {0};
    std::array<T, DIM> input_half_extent = {0};
    std::array<T, DIM> output_half_extent = {0};
  };

  void init(std::array<T, DIM> input_min, std::array<T, DIM> input_max,
            std::array<T, DIM> output_min, std::array<T, DIM> output_max) {
    for (IntType d = 0; d < DIM; ++d) {
      if(input_min[d] > input_max[d]) {
        throw InputError("Invalid input bounds.");
      }
      if (output_min[d] > output_max[d]) {
        throw InputError("Invalid output bounds.");
      }
    }

    for (IntType d = 0; d < DIM; ++d) {
      contrib::set_nhg_type3<T>(grid_info_.output_half_extent[d],
                                grid_info_.input_half_extent[d], opt_.upsampfac,
                                grid_info_.spread_grid_size[d],
                                &spread_grid_spacing_[d], &gam_[d]);
    }

    // reshape spread grid
    {
      bool new_grid = false;
      for (std::size_t d = 0; d < DIM; ++d) {
        new_grid |=
            (spread_grid_.shape(d) != grid_info_.padded_spread_grid_size[d]);
      }
      if (new_grid) {
        spread_grid_.reset(grid_info_.padded_spread_grid_size);
      }
    }

    // zero spread grid to prepare accumulation of input data
    spread_grid_.zero();

    // reshape fft grid
    {
      // create new grid if different
      bool new_grid = false;
      for (std::size_t d = 0; d < DIM; ++d) {
        new_grid |= (grid_info_.fft_grid_size[d] != fft_grid_.shape(d));
      }
      if (new_grid) {
        fft_grid_ = FFTGrid<T, DIM>(opt_.num_threads, grid_info_.fft_grid_size, sign_);
      }
    }

    // recompute correction factor for kernel windowing
    // we compute the inverse to use multiplication during execution
    for (std::size_t d = 0; d < DIM; ++d) {
      auto correction_fact_size = fft_grid_.shape(d) / 2 + 1;
      if (correction_factors_[d].shape(0) != correction_fact_size) {
        correction_factors_[d].reset(correction_fact_size);
      }

      contrib::onedim_fseries_kernel_inverse(
          fft_grid_.shape(d), correction_factors_[d].data(),
          kernel_param_.n_spread, kernel_param_.es_halfwidth,
          kernel_param_.es_beta, kernel_param_.es_c);
    }
  }

  Options opt_;
  ThreadPool th_pool_;
  HostArray<Point<T, DIM>, 1> rescaled_input_points_;
  HostArray<Point<T, DIM>, 1> rescaled_output_points_;
  std::vector<PartitionGroup> input_partition_;
  HostArray<std::complex<T>, 1> prephase_;
  HostArray<std::complex<T>, 1> postphase_;
  std::array<HostArray<T, 1>, DIM> correction_factors_;
  int sign_;
  KernelParameters<T> kernel_param_;
  FFTGrid<T, DIM> fft_grid_;
  GridInfo grid_info_;
  HostArray<std::complex<T>, DIM> spread_grid_;
  std::array<T, DIM> gam_;
  std::array<T, DIM> spread_grid_spacing_;
};

// TODO remove
template class PlanT3Impl<float, 1>;
template class PlanT3Impl<float, 2>;
template class PlanT3Impl<float, 3>;
template class PlanT3Impl<double, 1>;
template class PlanT3Impl<double, 2>;
template class PlanT3Impl<double, 3>;

} // namespace neonufft
