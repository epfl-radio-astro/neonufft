#pragma once

#include <cstddef>
#include <type_traits>

#include "neonufft/config.h"
// ---

#include "gpu/util/runtime_api.hpp"
#include "memory/view.hpp"
#include "neonufft/exceptions.hpp"
#include "neonufft/gpu/types.hpp"

namespace neonufft {
namespace gpu {

template <typename T, typename F, IntType DIM>
inline auto memcopy(ConstView<T, DIM> source, View<F, DIM> dest, const StreamType& stream) {
  static_assert(sizeof(T) == sizeof(F));

  if (source.size() == 0 && dest.size() == 0) return;

  if constexpr (DIM == 1) {
    if (source.shape(0) != dest.shape(0))
      throw InternalError("GPU view copy: shapes do not match.");
    // break recursion
    api::memcpy_async(dest.data(), source.data(), dest.size() * sizeof(T),
                      gpu::api::flag::MemcpyDefault, stream);

  } else {
    if (source.is_contiguous() && dest.is_contiguous()) {
      if (source.shape(0) != dest.shape(0))
        throw InternalError("GPU view copy: shapes do not match.");

      api::memcpy_async(dest.data(), source.data(), dest.size() * sizeof(T),
                        gpu::api::flag::MemcpyDefault, stream);
    } else {
      if constexpr (DIM == 2) {
        if (source.shape(1) != dest.shape(1))
          throw InternalError("GPU view copy: shapes do not match.");
        api::memcpy_2d_async(dest.data(), dest.strides(1) * sizeof(T), source.data(),
                             source.strides(1) * sizeof(T), dest.shape(0) * sizeof(T),
                             dest.shape(1), gpu::api::flag::MemcpyDefault, stream);
      } else {
        for (std::size_t i = 0; i < source.shape(DIM - 1); ++i) {
          memcopy<T, F, DIM - 1>(source.slice_view(i), dest.slice_view(i), stream);
        }
      }
    }
  }
}

// helper to auto convert to views
template <typename T, typename F>
inline auto memcopy(T&& source, F&& dest, const StreamType& stream) {
  static_assert(std::remove_reference_t<T>::dimension == std::remove_reference_t<F>::dimension);
  memcopy<typename std::remove_reference_t<T>::ValueType,
          typename std::remove_reference_t<F>::ValueType, std::remove_reference_t<T>::dimension>(
      source, dest, stream);
}

}  // namespace gpu
}  // namespace neonufft
