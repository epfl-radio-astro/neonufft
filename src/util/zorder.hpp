#pragma once

#include "neonufft/config.h"

#include <cstring>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "neonufft/types.hpp"

namespace neonufft {
namespace zorder {

template <typename T>
using KeyType =
    std::conditional_t<std::is_same_v<float, T>, std::uint32_t, std::uint64_t>;

template <typename T> inline auto convert_to_key(T x) -> KeyType<T> {
  static_assert(sizeof(T) == sizeof(KeyType<T>));

  KeyType<T> key;
  std::memcpy(&key, &x, sizeof(T));
  return key;
}

template <typename Key> inline bool less_msb(Key x, Key y) {
  return (x < y) && (x < (x ^ y));
}

// z-order comparison for float and double in [0, 1]
template <IntType DIM, typename T>
inline bool less_0_1(const T *p_lhs, const T *p_rhs) {
  static_assert(DIM > 0);
  static_assert(DIM <= 3);

  KeyType<T> lhs[DIM];
  KeyType<T> rhs[DIM];

  for (std::size_t i = 0; i < DIM; ++i) {
    assert(p_lhs[i] >= 0);
    assert(p_lhs[i] <= 1);
    assert(p_rhs[i] >= 0);
    assert(p_rhs[i] <= 1);
    lhs[i] = convert_to_key(p_lhs[i]);
    rhs[i] = convert_to_key(p_rhs[i]);
  }

  std::size_t ms_idx = 0;
  if constexpr (DIM > 1) {
    if (less_msb(lhs[0] ^ rhs[0], lhs[1] ^ rhs[1])) {
      ms_idx = 1;
    }
  }
  if constexpr (DIM > 2) {
    if (less_msb(lhs[ms_idx] ^ rhs[ms_idx], lhs[2] ^ rhs[2])) {
      ms_idx = 2;
    }
  }

  return lhs[ms_idx] < rhs[ms_idx];
}

} // namespace zorder

} // namespace neonufft
