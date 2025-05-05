#pragma once

#include <cstddef>
#include <cstring>

#include "neonufft/config.h"

#include "neonufft/exceptions.hpp"
#include "neonufft/memory/view.hpp"


namespace neonufft {

template <typename T, std::size_t DIM>
inline auto copy(const ConstHostView<T, DIM>& source, HostView<T, DIM> dest) {
  if (source.shape() != dest.shape())
    throw InternalError("Host view copy: shapes do not match.");

  if (source.size() == 0) return;

  if constexpr (DIM == 1) {
    std::memcpy(dest.data(), source.data(), source.shape() * sizeof(T));
  } else {
    for (std::size_t i = 0; i < source.shape()[DIM - 1]; ++i) {
      copy<T, DIM - 1>(source.slice_view(i), dest.slice_view(i));
    }
  }
}

template <typename T, std::size_t DIM>
inline auto copy(const HostView<T, DIM>& source, HostView<T, DIM> dest) {
  copy(ConstHostView<T, DIM>(source), std::move(dest));
}


}  // namespace neonufft
