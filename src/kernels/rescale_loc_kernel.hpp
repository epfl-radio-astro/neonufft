#pragma once

#include <complex>
#include <array>
#include <vector>

#include "neonufft/config.h"

#include "es_kernel_param.hpp"
#include "neonufft/types.hpp"
#include "util/partition_group.hpp"
#include "util/point.hpp"

namespace neonufft{
template <typename T, IntType DIM>
void rescale_loc(IntType n, std::array<const T *, DIM> loc, Point<T, DIM> *out);

template <typename T, IntType DIM>
void rescale_loc_t3(IntType n, std::array<T, DIM> offset,
                    std::array<T, DIM> scaling_factor,
                    std::array<const T *, DIM> loc, Point<T, DIM> *out);

template <typename T, IntType DIM>
std::vector<PartitionGroup>
rescale_loc_partition_t3(IntType partition_dim, IntType num_partitions,
                         IntType n, std::array<T, DIM> offset,
                         std::array<T, DIM> scaling_factor,
                         std::array<const T *, DIM> loc, Point<T, DIM> *out);
} // namespace neonufft
