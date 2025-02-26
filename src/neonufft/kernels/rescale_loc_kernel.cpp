#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <array>
#include <vector>

#include "neonufft/config.h"

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/enums.h"
#include "neonufft/exceptions.hpp"
#include "neonufft/types.hpp"
#include "neonufft/kernels/rescale_loc_kernel.hpp"
#include "neonufft/util/math.hpp"
// #include "neonufft/kernels/upsample_kernel.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "neonufft/kernels/rescale_loc_kernel.cpp" // this file

#include "neonufft/kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

// rescale values by first translating into [-pi, pi] and then rescaling to [0,
// 1]
template <typename T, IntType DIM>
HWY_ATTR void rescale_loc_kernel(IntType n, std::array<const T*, DIM> loc,
                                 Point<T, DIM> *HWY_RESTRICT out) {

  const TagType<T> d;
  const IntType n_lanes = hn::Lanes(d);

  const T two_pi = 2 * math::pi<T>;
  const T two_pi_inv = 1 / (2 * math::pi<T>);
  const T pi = math::pi<T>;

  const auto v_two_pi = hn::Set(d, two_pi);
  const auto v_two_pi_inv = hn::Set(d, two_pi_inv);
  const auto v_pi = hn::Set(d, pi);

  HWY_ALIGN std::array<T, n_lanes> buffer;

  IntType i = 0;

  for (; i + n_lanes <= n; i += n_lanes) {
    HWY_ALIGN std::array<Point<T, DIM>, n_lanes> points;
    for (IntType j = 0; j < n_lanes; ++j) {
      points[j].index = i + j;
    }
    for (IntType dim = 0; dim < DIM; ++dim) {
      auto l = hn::Mul(hn::Add(hn::LoadU(d, loc[dim] + i), v_pi), v_two_pi_inv);
      l = hn::Sub(l, hn::Floor(l));

      hn::Store(l, d, buffer.data());
      for (IntType j = 0; j < n_lanes; ++j) {
        points[j].coord[dim] = buffer[j];
      }
    }
    std::memcpy(out + i, points.data(), n_lanes * sizeof(Point<T, DIM>));
  }

  for (; i < n; ++i) {
    Point<T, DIM> p;
    p.index = i;

    for (IntType dim = 0; dim < DIM; ++dim) {
      auto l = (loc[dim][i] + pi) * two_pi_inv;
      l = l - std::floor(l);
      p.coord[dim] = l;
    }

    out[i] = p;
  }
}

// rescale values after offsetting and rescaling, translating into [-pi, pi] and
// then rescaling to [0, 1]
template <typename T, IntType DIM>
HWY_ATTR void rescale_loc_t3_kernel(IntType n, std::array<T, DIM> offset,
                                    std::array<T, DIM> scaling_factor,
                                    std::array<const T *, DIM> loc,
                                    Point<T, DIM> *HWY_RESTRICT out) {

  const TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);

  const T two_pi = 2 * math::pi<T>;
  const T two_pi_inv = 1 / (2 * math::pi<T>);
  const T pi = math::pi<T>;

  const auto v_two_pi = hn::Set(d, two_pi);
  const auto v_two_pi_inv = hn::Set(d, two_pi_inv);
  const auto v_pi = hn::Set(d, pi);

  HWY_ALIGN std::array<T, n_lanes> buffer;

  IntType i = 0;

  HWY_ALIGN std::array<Point<T, DIM>, n_lanes> points;
  for (; i + n_lanes <= n; i += n_lanes) {
    for (IntType j = 0; j < n_lanes; ++j) {
      points[j].index = i + j;
    }
    for (IntType dim = 0; dim < DIM; ++dim) {
      const auto v_offset = hn::Set(d, offset[dim]);
      const auto v_scaling_factor = hn::Set(d, scaling_factor[dim]);
      auto l = hn::LoadU(d, loc[dim] + i);
      l = hn::Mul(hn::Sub(l, v_offset), v_scaling_factor);
      l = hn::Mul(hn::Add(l, v_pi), v_two_pi_inv);
      l = hn::Sub(l, hn::Floor(l));

      hn::Store(l, d, buffer.data());
      for (IntType j = 0; j < n_lanes; ++j) {
        points[j].coord[dim] = T(buffer[j]);
      }
    }
    std::memcpy(out + i, points.data(), n_lanes * sizeof(Point<T, DIM>));
  }

  for (; i < n; ++i) {
    Point<T, DIM> p;
    p.index = i;

    for (IntType dim = 0; dim < DIM; ++dim) {
      T l = (loc[dim][i] - offset[dim]) * scaling_factor[dim];
      l = (l + pi) * two_pi_inv;
      l = l - std::floor(l);
      p.coord[dim] = l;
    }

    out[i] = p;
  }
}

template <typename T, IntType DIM>
HWY_ATTR std::vector<PartitionGroup> rescale_loc_partition_t3_kernel(IntType partition_dim,
                                              IntType num_partitions, IntType n,
                                              std::array<T, DIM> offset,
                                              std::array<T, DIM> scaling_factor,
                                              std::array<const T *, DIM> loc,
                                              Point<T, DIM> *HWY_RESTRICT out) {

  if (partition_dim >= DIM || partition_dim < 0) {
    throw InternalError("Invalid partition dimension");
  }

  if (num_partitions < 1) {
    throw InternalError("Invalid number of partitions");
  }

  const TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);

  const T two_pi = 2 * math::pi<T>;
  const T two_pi_inv = 1 / (2 * math::pi<T>);
  const T pi = math::pi<T>;

  const auto v_two_pi = hn::Set(d, two_pi);
  const auto v_two_pi_inv = hn::Set(d, two_pi_inv);
  const auto v_pi = hn::Set(d, pi);

  HWY_ALIGN std::array<T, n_lanes> buffer;
  HWY_ALIGN std::array<Point<T, DIM>, n_lanes> points;

  std::vector<PartitionGroup> groups(num_partitions);

  // compute size of partition groups
  {
    IntType i = 0;
    for (; i + n_lanes <= n; i += n_lanes) {
      const auto v_offset = hn::Set(d, offset[partition_dim]);
      const auto v_scaling_factor = hn::Set(d, scaling_factor[partition_dim]);
      auto l = hn::LoadU(d, loc[partition_dim] + i);
      l = hn::Mul(hn::Sub(l, v_offset), v_scaling_factor);
      l = hn::Mul(hn::Add(l, v_pi), v_two_pi_inv);
      l = hn::Sub(l, hn::Floor(l));

      hn::Store(l, d, buffer.data());
      for (IntType j = 0; j < n_lanes; ++j) {
        const IntType idx_group =
            std::min<IntType>(num_partitions - 1, buffer[j] * num_partitions);
        ++groups[idx_group].size;
      }
    }

    for (; i < n; ++i) {
      T l = (loc[partition_dim][i] - offset[partition_dim]) *
            scaling_factor[partition_dim];
      l = (l + pi) * two_pi_inv;
      l = l - std::floor(l);

      const IntType idx_group =
          std::min<IntType>(num_partitions - 1, l * num_partitions);
      ++groups[idx_group].size;
    }
  }

  // assign group begin indices
  for (IntType idx_group = 1; idx_group < num_partitions; ++idx_group) {
    groups[idx_group].begin =
        groups[idx_group - 1].begin + groups[idx_group - 1].size;
  }

  // create actual points and write to assigned group
  {
    std::array<IntType, n_lanes> group_indices;
    IntType i = 0;
    for (; i + n_lanes <= n; i += n_lanes) {
      for (IntType j = 0; j < n_lanes; ++j) {
        points[j].index = i + j;
      }
      for (IntType dim = 0; dim < DIM; ++dim) {
        const auto v_offset = hn::Set(d, offset[dim]);
        const auto v_scaling_factor = hn::Set(d, scaling_factor[dim]);
        auto l = hn::LoadU(d, loc[dim] + i);
        l = hn::Mul(hn::Sub(l, v_offset), v_scaling_factor);
        l = hn::Mul(hn::Add(l, v_pi), v_two_pi_inv);
        l = hn::Sub(l, hn::Floor(l));

        hn::Store(l, d, buffer.data());
        for (IntType j = 0; j < n_lanes; ++j) {
          points[j].coord[dim] = buffer[j];
        }
        if (dim == partition_dim) {
          for (IntType j = 0; j < n_lanes; ++j) {
            group_indices[j] = std::min<IntType>(num_partitions - 1,
                                                 buffer[j] * num_partitions);
          }
        }
      }

      // copy points to permuted locations
      for (IntType j = 0; j < n_lanes; ++j) {
        out[groups[group_indices[j]].begin++] = points[j];
      }
    }

    for (; i < n; ++i) {
      Point<T, DIM> p;
      p.index = i;

      IntType idx_group = 0;
      for (IntType dim = 0; dim < DIM; ++dim) {
        T l = (loc[dim][i] - offset[dim]) * scaling_factor[dim];
        l = (l + pi) * two_pi_inv;
        l = l - std::floor(l);
        p.coord[dim] = l;
        if (dim == partition_dim) {
          idx_group = std::min<IntType>(num_partitions - 1, l * num_partitions);
        }
      }

      out[groups[idx_group].begin++] = p;
    }
  }

  // reset group begin indices
  for (auto &g : groups) {
    g.begin -= g.size;
  }

#ifndef NDEBUG
// sanity check
  IntType sum = 0;
  for(IntType idx = 0; idx < n; ++idx) {
    sum += out[idx].index;
  }
  assert(sum == (n * (n - 1)) / 2);
#endif

  return groups;
}

template <IntType DIM>
HWY_ATTR void rescale_loc_kernel_float(IntType n,
                                       std::array<const float *, DIM> loc,
                                       Point<float, DIM> *out) {
  rescale_loc_kernel<float, DIM>(n, {loc}, out);
}

template <IntType DIM>
HWY_ATTR void rescale_loc_kernel_double(IntType n,
                                        std::array<const double *, DIM> loc,
                                        Point<double, DIM> *out) {
  rescale_loc_kernel<double, DIM>(n, {loc}, out);
}

template <IntType DIM>
HWY_ATTR void rescale_loc_t3_kernel_float(IntType n,
                                          std::array<float, DIM> offset,
                                          std::array<float, DIM> scaling_factor,
                                          std::array<const float *, DIM> loc,
                                          Point<float, DIM> *out) {
  rescale_loc_t3_kernel<float, DIM>(n, offset, scaling_factor, loc, out);
}

template <IntType DIM>
HWY_ATTR void
rescale_loc_t3_kernel_double(IntType n, std::array<double, DIM> offset,
                             std::array<double, DIM> scaling_factor,
                             std::array<const double *, DIM> loc,
                             Point<double, DIM> *out) {
  rescale_loc_t3_kernel<double, DIM>(n, offset, scaling_factor, loc, out);
}

template <IntType DIM>
HWY_ATTR  void rescale_loc_partition_t3_float(
    IntType partition_dim, IntType num_partitions, IntType n,
    std::array<float, DIM> offset, std::array<float, DIM> scaling_factor,
    std::array<const float *, DIM> loc, Point<float, DIM> *HWY_RESTRICT out, std::vector<PartitionGroup>& groups) {
  groups = rescale_loc_partition_t3_kernel<float, DIM>(
      partition_dim, num_partitions, n, offset, scaling_factor, loc, out);
}

template <IntType DIM>
HWY_ATTR void rescale_loc_partition_t3_double(IntType partition_dim,
                                         IntType num_partitions, IntType n,
                                         std::array<double, DIM> offset,
                                         std::array<double, DIM> scaling_factor,
                                         std::array<const double *, DIM> loc,
                                         Point<double, DIM> *HWY_RESTRICT out,
                                         std::vector<PartitionGroup> &groups) {
  groups = rescale_loc_partition_t3_kernel<double, DIM>(
      partition_dim, num_partitions, n, offset, scaling_factor, loc, out);
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void rescale_loc(IntType n, std::array<const T *, DIM> loc, Point<T, DIM> *out) {
  if constexpr(std::is_same_v<float, T>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(rescale_loc_kernel_float<DIM>)
    (n, loc, out);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(rescale_loc_kernel_double<DIM>)
    (n, loc, out);
  }
}

template void rescale_loc<float, 1>(IntType n, std::array<const float *, 1> loc,
                                    Point<float, 1> *out);

template void rescale_loc<float, 2>(IntType n, std::array<const float *, 2> loc,
                                    Point<float, 2> *out);

template void rescale_loc<float, 3>(IntType n, std::array<const float *, 3> loc,
                                    Point<float, 3> *out);

template void rescale_loc<double, 1>(IntType n, std::array<const double *, 1> loc,
                                    Point<double, 1> *out);

template void rescale_loc<double, 2>(IntType n, std::array<const double *, 2> loc,
                                    Point<double, 2> *out);

template void rescale_loc<double, 3>(IntType n, std::array<const double *, 3> loc,
                                    Point<double, 3> *out);

template <typename T, IntType DIM>
void rescale_loc_t3(IntType n, std::array<T, DIM> offset,
                    std::array<T, DIM> scaling_factor,
                    std::array<const T *, DIM> loc, Point<T, DIM> *out) {
  if constexpr(std::is_same_v<float, T>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(rescale_loc_t3_kernel_float<DIM>)
    (n, offset, scaling_factor, loc, out);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(rescale_loc_t3_kernel_double<DIM>)
    (n, offset, scaling_factor, loc, out);
  }
}

template void rescale_loc_t3<float, 1>(IntType n, std::array<float, 1> offset,
                                       std::array<float, 1> scaling_factor,
                                       std::array<const float *, 1> loc,
                                       Point<float, 1> *out);

template void rescale_loc_t3<float, 2>(IntType n, std::array<float, 2> offset,
                                       std::array<float, 2> scaling_factor,
                                       std::array<const float *, 2> loc,
                                       Point<float, 2> *out);

template void rescale_loc_t3<float, 3>(IntType n, std::array<float, 3> offset,
                                       std::array<float, 3> scaling_factor,
                                       std::array<const float *, 3> loc,
                                       Point<float, 3> *out);

template void rescale_loc_t3<double, 1>(IntType n, std::array<double, 1> offset,
                                        std::array<double, 1> scaling_factor,
                                        std::array<const double *, 1> loc,
                                        Point<double, 1> *out);

template void rescale_loc_t3<double, 2>(IntType n, std::array<double, 2> offset,
                                        std::array<double, 2> scaling_factor,
                                        std::array<const double *, 2> loc,
                                        Point<double, 2> *out);

template void rescale_loc_t3<double, 3>(IntType n, std::array<double, 3> offset,
                                        std::array<double, 3> scaling_factor,
                                        std::array<const double *, 3> loc,
                                        Point<double, 3> *out);

template <typename T, IntType DIM>
std::vector<PartitionGroup> rescale_loc_partition_t3(IntType partition_dim, IntType num_partitions,
                              IntType n, std::array<T, DIM> offset,
                              std::array<T, DIM> scaling_factor,
                              std::array<const T *, DIM> loc,
                              Point<T, DIM> *out) {
  std::vector<PartitionGroup> groups;
  if constexpr (std::is_same_v<float, T>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(rescale_loc_partition_t3_float<DIM>)
    (partition_dim, num_partitions, n, offset, scaling_factor, loc, out,
     groups);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(rescale_loc_partition_t3_double<DIM>)
    (partition_dim, num_partitions, n, offset, scaling_factor, loc, out,
     groups);
  }
  return groups;
}

template std::vector<PartitionGroup> rescale_loc_partition_t3<float, 1>(
    IntType partition_dim, IntType num_partitions, IntType n,
    std::array<float, 1> offset, std::array<float, 1> scaling_factor,
    std::array<const float *, 1> loc, Point<float, 1> *out);

template std::vector<PartitionGroup> rescale_loc_partition_t3<float, 2>(
    IntType partition_dim, IntType num_partitions, IntType n,
    std::array<float, 2> offset, std::array<float, 2> scaling_factor,
    std::array<const float *, 2> loc, Point<float, 2> *out);

template std::vector<PartitionGroup> rescale_loc_partition_t3<float, 3>(
    IntType partition_dim, IntType num_partitions, IntType n,
    std::array<float, 3> offset, std::array<float, 3> scaling_factor,
    std::array<const float *, 3> loc, Point<float, 3> *out);

template std::vector<PartitionGroup> rescale_loc_partition_t3<double, 1>(
    IntType partition_dim, IntType num_partitions, IntType n,
    std::array<double, 1> offset, std::array<double, 1> scaling_factor,
    std::array<const double *, 1> loc, Point<double, 1> *out);

template std::vector<PartitionGroup> rescale_loc_partition_t3<double, 2>(
    IntType partition_dim, IntType num_partitions, IntType n,
    std::array<double, 2> offset, std::array<double, 2> scaling_factor,
    std::array<const double *, 2> loc, Point<double, 2> *out);

template std::vector<PartitionGroup> rescale_loc_partition_t3<double, 3>(
    IntType partition_dim, IntType num_partitions, IntType n,
    std::array<double, 3> offset, std::array<double, 3> scaling_factor,
    std::array<const double *, 3> loc, Point<double, 3> *out);

#endif

} // namespace neonufft
