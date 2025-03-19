#pragma once

#include "neonufft/config.h"
// ---

#include <array>
#include <cassert>
#include <type_traits>

#include "neonufft/types.hpp"
#include "neonufft/util/func_attributes.hpp"

namespace neonufft {
template<typename T, IntType DIM>
struct StackArray {
  StackArray() = default;

  NEONUFFT_H_FUNC StackArray(const std::array<T, DIM>& a) noexcept(
      std::is_nothrow_copy_assignable_v<T>) {
    for (IntType d = 0; d < DIM; ++d) {
      values[d] = a[d];
    }
  }

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) const noexcept -> const T& {
    assert(index < DIM);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) noexcept -> T& {
    assert(index < DIM);
    return values[index];
  }

  T values[DIM];
};

}  // namespace neonufft
