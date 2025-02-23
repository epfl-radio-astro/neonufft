#pragma once

#include "neonufft/config.h"
// ---

#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "gpu/memory/device_array.hpp"
#include "neonufft/allocator.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/types.hpp"

namespace neonufft {
namespace gpu {

template <typename T, std::size_t DIM> class FFTGrid {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM >= 1 && DIM <= 3);

  using IndexType = typename DeviceArray<ComplexType<T>, DIM>::IndexType;

  FFTGrid();

  FFTGrid(const std::shared_ptr<Allocator>& alloc, StreamType stream, IndexType shape, int sign,
          IndexType padding = IndexType());

  FFTGrid(const FFTGrid&) = delete;

  FFTGrid(FFTGrid&&) noexcept = default;

  auto operator=(const FFTGrid&) -> FFTGrid& = delete;

  auto operator=(FFTGrid&& b) noexcept -> FFTGrid& = default;

  DeviceView<ComplexType<T>, DIM> view() { return grid_; }

  DeviceView<ComplexType<T>, DIM> padded_view() { return padded_grid_.view(); }

  IndexType shape() { return grid_.shape(); }

  IntType shape(IntType i) { return grid_.shape(i); }

  IndexType padding() { return padding_; }

  void transform();

private:
  std::unique_ptr<void, void (*)(void *)> plan_;
  IndexType padding_;
  DeviceArray<ComplexType<T>, DIM> padded_grid_;
  DeviceArray<char, 1> workspace_;
  DeviceView<ComplexType<T>, DIM> grid_;
  int sign_ = -1;
};

}  // namespace gpu
} // namespace neonufft
