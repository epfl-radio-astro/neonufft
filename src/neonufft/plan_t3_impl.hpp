#pragma once

#include "neonufft/config.h"
//--

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>

#include "contrib/es_kernel/util.hpp"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/exceptions.hpp"
#include "neonufft/kernels/compute_postphase_kernel.hpp"
#include "neonufft/kernels/compute_prephase_kernel.hpp"
#include "neonufft/kernels/fold_padding.hpp"
#include "neonufft/kernels/fseries_kernel.hpp"
#include "neonufft/kernels/interpolation_kernel.hpp"
#include "neonufft/kernels/min_max_kernel.hpp"
#include "neonufft/kernels/nuft_real_kernel.hpp"
#include "neonufft/kernels/rescale_loc_kernel.hpp"
#include "neonufft/kernels/spreading_kernel.hpp"
#include "neonufft/kernels/upsample_kernel.hpp"
#include "neonufft/memory/array.hpp"
#include "neonufft/memory/copy.hpp"
#include "neonufft/memory/view.hpp"
#include "neonufft/plan.hpp"
#include "neonufft/threading//thread_pool.hpp"
#include "neonufft/types.hpp"
#include "neonufft/util/fft_grid.hpp"
#include "neonufft/util/grid_info_t3.hpp"
#include "neonufft/util/point.hpp"
#include "neonufft/util/spread_padding.hpp"
#include "neonufft/util/zorder.hpp"

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
    GridInfoT3<T, DIM> grid_info(kernel_param.n_spread, opt.upsampfac,
                       opt.recenter_threshold, input_min, input_max, output_min,
                       output_max);

    const std::uint64_t total_spread_grid_size = std::reduce(
        grid_info.padded_spread_grid_size.begin(),
        grid_info.padded_spread_grid_size.end(), 1, std::multiplies<IntType>());

    const std::uint64_t total_fft_grid_size = std::reduce(
        grid_info.fft_grid_size.begin(), grid_info.fft_grid_size.end(), 1,
        std::multiplies<IntType>());

    return sizeof(std::complex<T>) * (total_spread_grid_size + total_fft_grid_size);
  }

  PlanT3Impl(IntType batch_size, Options opt, int sign, IntType num_in,
             std::array<const T*, DIM> input_points, IntType num_out,
             std::array<const T*, DIM> output_points)
      : batch_size_(batch_size),
        opt_(opt),
        th_pool_(opt_.num_threads),
        sign_(sign),
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

    grid_info_ = GridInfoT3<T, DIM>(kernel_param_.n_spread, opt_.upsampfac,
                          opt_.recenter_threshold, input_min, input_max,
                          output_min, output_max);

    this->init(input_min, input_max, output_min, output_max);
    this->set_input_points(num_in, input_points);
    this->set_output_points(num_out, output_points);
  }

  PlanT3Impl(IntType batch_size, Options opt, int sign, std::array<T, DIM> input_min,
             std::array<T, DIM> input_max, std::array<T, DIM> output_min,
             std::array<T, DIM> output_max)
      : batch_size_(batch_size),
        opt_(opt),
        th_pool_(opt_.num_threads),
        sign_(sign),
        kernel_param_(opt_.tol, opt_.upsampfac, opt_.kernel_approximation),
        grid_info_(kernel_param_.n_spread, opt_.upsampfac, opt_.recenter_threshold, input_min,
                   input_max, output_min, output_max) {
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

  void transform(std::complex<T>* out, IntType bdist) {
    if (bdist <= 0) bdist = rescaled_output_points_.shape(0);

    std::array<ConstHostView<T, 1>, DIM> correction_factor_views;
    for (IntType dim = 0; dim < DIM; ++dim) {
      correction_factor_views[dim] = correction_factors_[dim].view();
    }

    IndexArray<DIM> padding;
    padding.fill(spread_padding(kernel_param_.n_spread));

    for (IntType idx_batch = 0; idx_batch < batch_size_; ++idx_batch) {
      // TODO: move to init
      fft_grid_.padded_view().zero();

      std::complex<T>* out_batch = out + idx_batch * bdist;

      // TODO: folding might not be needed. Do we ever have points on the actual
      // edges? check how points are rescaled. Unit tests show values in padding
      // is always 0
      // fold_padding<T, DIM>(kernel_param_.n_spread, spread_grid_);

      upsample<T, DIM>(
          NEONUFFT_MODE_ORDER_CMCL,
          spread_grid_.slice_view(idx_batch).sub_view(padding, grid_info_.spread_grid_size),
          correction_factor_views, fft_grid_.view());

      fft_grid_.transform();

      if (th_pool_.num_threads() > 1) {
        th_pool_.parallel_for({0, rescaled_output_points_.size()}, [&](IntType, BlockRange range) {
          auto sub_view = rescaled_output_points_.sub_view(range.begin, range.end - range.begin);
          interpolate<T, DIM>(opt_.kernel_type, kernel_param_, fft_grid_.view(), sub_view.shape(0),
                              sub_view.data(), out_batch);
        });
      } else {
        interpolate<T, DIM>(opt_.kernel_type, kernel_param_, fft_grid_.view(),
                            rescaled_output_points_.shape(0), rescaled_output_points_.data(),
                            out_batch);
      }
      // TODO: move to kernel. NOTE: integration into interpolate functions
      // results in large performance hit with clang (18)
      for (IntType i = 0; i < postphase_.shape(0); ++i) {
        out_batch[i] *= postphase_[i];
      }
    }

    // Reset spread grid
    spread_grid_.zero();
  }

  void add_input(const std::complex<T>* in, IntType bdist) {
    if (bdist <= 0) bdist = rescaled_input_points_.shape(0);

    for (IntType idx_batch = 0; idx_batch < batch_size_; ++idx_batch) {
      const std::complex<T>* in_batch = in + idx_batch * bdist;

      auto spread_grid_batch = spread_grid_.slice_view(idx_batch);

      if (th_pool_.num_threads() > 1) {
        // spread even groups
        th_pool_.parallel_for(
            {0, IntType(input_partition_.size())}, 2, [&](IntType, BlockRange range) {
              for (IntType idx_group = range.begin; idx_group < range.end; ++idx_group) {
                const auto& g = input_partition_[idx_group];
                if (idx_group % 2 == 0 && g.size) {
                  auto sub_view = rescaled_input_points_.sub_view(g.begin, g.size);
                  spread<T, DIM>(opt_.kernel_type, kernel_param_, sub_view.shape(0),
                                 sub_view.data(), in_batch,
                                 prephase_.size() ? prephase_.data() : nullptr,
                                 grid_info_.spread_grid_size, spread_grid_batch);
                }
              }
            });

        // spread odd groups
        th_pool_.parallel_for(
            {0, IntType(input_partition_.size())}, 2, [&](IntType, BlockRange range) {
              for (IntType idx_group = range.begin; idx_group < range.end; ++idx_group) {
                const auto& g = input_partition_[idx_group];
                if (idx_group % 2 == 1 && g.size) {
                  auto sub_view = rescaled_input_points_.sub_view(g.begin, g.size);
                  spread<T, DIM>(opt_.kernel_type, kernel_param_, sub_view.shape(0),
                                 sub_view.data(), in_batch,
                                 prephase_.size() ? prephase_.data() : nullptr,
                                 grid_info_.spread_grid_size, spread_grid_batch);
                }
              }
            });
      } else {
        spread<T, DIM>(opt_.kernel_type, kernel_param_, rescaled_input_points_.shape(0),
                       rescaled_input_points_.data(), in_batch,
                       prephase_.size() ? prephase_.data() : nullptr, grid_info_.spread_grid_size,
                       spread_grid_batch);
      }
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
      batched_shape[d] = grid_info_.padded_spread_grid_size[d];
    }
    spread_grid_.reset(batched_shape);

    // zero spread grid to prepare accumulation of input data
    spread_grid_.zero();

    // reshape fft grid
    fft_grid_ = FFTGrid<T, DIM>(opt_.num_threads, grid_info_.fft_grid_size, sign_);

    // recompute correction factor for kernel windowing
    // we compute the inverse to use multiplication during execution
    for (std::size_t d = 0; d < DIM; ++d) {
      auto correction_fact_size = fft_grid_.shape(d) / 2 + 1;
      if (correction_factors_[d].shape(0) != correction_fact_size) {
        correction_factors_[d].reset(correction_fact_size);
      }

      fseries_inverse<T>(opt_.kernel_type, kernel_param_, fft_grid_.shape(d),
                         correction_factors_[d].data());
    }
  }

  IntType batch_size_;
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
  GridInfoT3<T, DIM> grid_info_;
  HostArray<std::complex<T>, DIM + 1> spread_grid_;
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

}  // namespace neonufft
