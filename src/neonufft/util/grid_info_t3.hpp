#pragma once

#include "neonufft/config.h"
//---

#include <array>
#include <cmath>

#include "neonufft/types.hpp"
#include "contrib/es_kernel/util.hpp"
#include "neonufft/util/spread_padding.hpp"

namespace neonufft {
template <typename T, IntType DIM>
struct GridInfoT3 {
  GridInfoT3() = default;

  GridInfoT3(IntType n_spread, double upsampfac, T recenter_threshold,
             std::array<T, DIM> input_min_i, std::array<T, DIM> input_max_i,
             std::array<T, DIM> output_min_i, std::array<T, DIM> output_max_i)
      : input_min(input_min_i),
        input_max(input_max_i),
        output_min(output_min_i),
        output_max(output_max_i) {
    for (IntType d = 0; d < DIM; ++d) {
      input_offsets[d] = (input_max[d] + input_min[d]) / 2;
      output_offsets[d] = (output_max[d] + output_min[d]) / 2;

      input_half_extent[d] = std::max(std::abs(input_max[d] - input_offsets[d]),
                                      std::abs(input_min[d] - input_offsets[d]));

      output_half_extent[d] = std::max(std::abs(output_max[d] - output_offsets[d]),
                                       std::abs(output_min[d] - output_offsets[d]));

      if (std::abs(input_offsets[d]) < recenter_threshold * input_half_extent[d]) {
        input_offsets[d] = 0;
        input_half_extent[d] = std::max(std::abs(input_max[d]), std::abs(input_min[d]));
      }
      if (std::abs(output_offsets[d]) < recenter_threshold * output_half_extent[d]) {
        output_offsets[d] = 0;
        output_half_extent[d] = std::max(std::abs(output_max[d]), std::abs(output_min[d]));
      }

      spread_grid_size[d] = contrib::t3_grid_size<T>(output_half_extent[d], input_half_extent[d],
                                                     upsampfac, n_spread);

      padded_spread_grid_size[d] = spread_grid_size[d] + 2 * spread_padding(n_spread);

      fft_grid_size[d] = contrib::next235even(
          std::max<std::size_t>(2 * n_spread, spread_grid_size[d] * upsampfac));
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

}  // namespace neonufft
