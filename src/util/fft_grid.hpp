#pragma once

#include "neonufft/config.h"

#include "neonufft/types.hpp"
#include "memory/array.hpp"

#include <any>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <memory>

namespace neonufft {

template <typename T, std::size_t DIM> class FFTGrid {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM >= 1 && DIM <= 3);

  using IndexType = typename HostArray<std::complex<T>, DIM>::IndexType;

  FFTGrid();

  FFTGrid(IntType num_threads, IndexType shape, int sign,
          IndexType padding = IndexType());

  FFTGrid(const FFTGrid&) = delete;

  FFTGrid(FFTGrid&&) noexcept = default;

  auto operator=(const FFTGrid&) -> FFTGrid& = delete;

  auto operator=(FFTGrid&& b) noexcept -> FFTGrid& = default;

  HostView<std::complex<T>, DIM> view() { return grid_; }

  HostView<std::complex<T>, DIM> padded_view() { return padded_grid_.view(); }

  IndexType shape() { return grid_.shape(); }

  IntType shape(IntType i) { return grid_.shape(i); }

  IndexType padding() { return padding_; }

  void transform();

private:
  std::unique_ptr<void, void (*)(void *)> plan_;
  IndexType padding_;
  HostArray<std::complex<T>, DIM> padded_grid_;
  HostView<std::complex<T>, DIM> grid_;
};

} // namespace neonufft
