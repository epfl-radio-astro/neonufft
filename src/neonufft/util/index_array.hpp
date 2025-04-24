#pragma once

#include "neonufft/config.h"
// ---

#include <cassert>
#include <array>

#include "neonufft/types.hpp"
#include "neonufft/util/func_attributes.hpp"

namespace neonufft {
template <IntType DIM>
struct IndexArray;

template <>
struct IndexArray<0> {};

template <>
struct IndexArray<1> {
  static inline constexpr IntType DIM = 1;

  NEONUFFT_H_D_FUNC IndexArray() noexcept : values{0} {}

  NEONUFFT_H_D_FUNC IndexArray(IntType idx) noexcept : values{idx} {}

  NEONUFFT_H_FUNC IndexArray(std::array<IntType, 1> idx) noexcept : values{idx[0]} {}

  NEONUFFT_H_FUNC operator std::array<IntType, 1>() const {
    return std::array<IntType, 1>{values[0]};
  };

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) const noexcept -> const IntType& {
    assert(index < 1);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) noexcept -> IntType& {
    assert(index < 1);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto fill(const IntType& val) noexcept -> void { values[0] = val; }

  NEONUFFT_H_D_FUNC inline auto begin() noexcept -> IntType* { return values; }
  NEONUFFT_H_D_FUNC inline auto begin() const noexcept -> const IntType* { return values; }

  NEONUFFT_H_D_FUNC inline auto end() noexcept -> IntType* { return values + DIM; }
  NEONUFFT_H_D_FUNC inline auto end() const noexcept -> const IntType* { return values + DIM; }

  NEONUFFT_H_D_FUNC inline auto size() const noexcept -> IntType { return DIM; }

  NEONUFFT_H_FUNC inline auto to_array() const noexcept -> std::array<IntType, DIM> {
    return std::array<IntType, DIM>{values[0]};
  }

  IntType values[1] = {0};
};

template <>
struct IndexArray<2> {
  static inline constexpr IntType DIM = 2;

  NEONUFFT_H_D_FUNC IndexArray() noexcept : values{0, 0} {}

  NEONUFFT_H_D_FUNC IndexArray(IntType idx0, IntType idx1) noexcept : values{idx0, idx1} {}

  NEONUFFT_H_FUNC IndexArray(std::array<IntType, DIM> idx) noexcept : values{idx[0], idx[1]} {}

  NEONUFFT_H_FUNC operator std::array<IntType, 2>() const {
    return std::array<IntType, 2>{values[0], values[1]};
  };

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) const noexcept -> const IntType& {
    assert(index < DIM);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) noexcept -> IntType& {
    assert(index < DIM);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto fill(const IntType& val) noexcept -> void {
    values[0] = val;
    values[1] = val;
  }

  NEONUFFT_H_D_FUNC inline auto begin() noexcept -> IntType* { return values; }
  NEONUFFT_H_D_FUNC inline auto begin() const noexcept -> const IntType* { return values; }

  NEONUFFT_H_D_FUNC inline auto end() noexcept -> IntType* { return values + DIM; }
  NEONUFFT_H_D_FUNC inline auto end() const noexcept -> const IntType* { return values + DIM; }

  NEONUFFT_H_D_FUNC inline auto size() const noexcept -> IntType { return DIM; }

  NEONUFFT_H_FUNC inline auto to_array() const noexcept -> std::array<IntType, DIM> {
    return std::array<IntType, DIM>{values[0], values[1]};
  }

  IntType values[DIM] = {0};
};

template <>
struct IndexArray<3> {
  static inline constexpr IntType DIM = 3;

  NEONUFFT_H_D_FUNC IndexArray() noexcept : values{0, 0, 0} {}

  NEONUFFT_H_D_FUNC IndexArray(IntType idx0, IntType idx1, IntType idx2) noexcept
      : values{idx0, idx1, idx2} {}

  NEONUFFT_H_FUNC IndexArray(std::array<IntType, DIM> idx) noexcept
      : values{idx[0], idx[1], idx[2]} {}

  NEONUFFT_H_FUNC operator std::array<IntType, 3>() const {
    return std::array<IntType, 3>{values[0], values[1], values[2]};
  };

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) const noexcept -> const IntType& {
    assert(index < DIM);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) noexcept -> IntType& {
    assert(index < DIM);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto fill(const IntType& val) noexcept -> void {
    values[0] = val;
    values[1] = val;
    values[2] = val;
  }

  NEONUFFT_H_D_FUNC inline auto begin() noexcept -> IntType* { return values; }
  NEONUFFT_H_D_FUNC inline auto begin() const noexcept -> const IntType* { return values; }

  NEONUFFT_H_D_FUNC inline auto end() noexcept -> IntType* { return values + DIM; }
  NEONUFFT_H_D_FUNC inline auto end() const noexcept -> const IntType* { return values + DIM; }

  NEONUFFT_H_D_FUNC inline auto size() const noexcept -> IntType { return DIM; }

  NEONUFFT_H_FUNC inline auto to_array() const noexcept -> std::array<IntType, DIM> {
    return std::array<IntType, DIM>{values[0], values[1], values[2]};
  }

  IntType values[DIM] = {0};
};

template <>
struct IndexArray<4> {
  static inline constexpr IntType DIM = 4;

  NEONUFFT_H_D_FUNC IndexArray() noexcept : values{0, 0, 0, 0} {}

  NEONUFFT_H_D_FUNC IndexArray(IntType idx0, IntType idx1, IntType idx2, IntType idx3) noexcept
      : values{idx0, idx1, idx2, idx3} {}

  NEONUFFT_H_FUNC IndexArray(std::array<IntType, DIM> idx) noexcept
      : values{idx[0], idx[1], idx[2], idx[3]} {}

  NEONUFFT_H_FUNC operator std::array<IntType, 4>() const {
    return std::array<IntType, 4>{values[0], values[1], values[2], values[3]};
  };

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) const noexcept -> const IntType& {
    assert(index < DIM);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto operator[](const IntType& index) noexcept -> IntType& {
    assert(index < DIM);
    return values[index];
  }

  NEONUFFT_H_D_FUNC inline auto fill(const IntType& val) noexcept -> void {
    values[0] = val;
    values[1] = val;
    values[2] = val;
    values[3] = val;
  }

  NEONUFFT_H_D_FUNC inline auto begin() noexcept -> IntType* { return values; }
  NEONUFFT_H_D_FUNC inline auto begin() const noexcept -> const IntType* { return values; }

  NEONUFFT_H_D_FUNC inline auto end() noexcept -> IntType* { return values + DIM; }
  NEONUFFT_H_D_FUNC inline auto end() const noexcept -> const IntType* { return values + DIM; }

  NEONUFFT_H_D_FUNC inline auto size() const noexcept -> IntType { return DIM; }

  NEONUFFT_H_FUNC inline auto to_array() const noexcept -> std::array<IntType, DIM> {
    return std::array<IntType, DIM>{values[0], values[1], values[2], values[3]};
  }

  IntType values[DIM] = {0};
};


}  // namespace neonufft
