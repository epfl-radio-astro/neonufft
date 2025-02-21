#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>

#include "neonufft/config.h"

#include "memory/view.hpp"

#ifndef NEONUFFT_ALIGNMENT
// covers all vector sizes (128 * 8 bits) and cache line sizes.
#define NEONUFFT_ALIGNMENT 128
#define NEONUFFT_MAX_VEC_LENGTH 128 // must be multiple of NEONUFFT_ALIGNMENT
#endif

/*
 * Arrays are views, that own the attached memory.
 */

namespace neonufft {

namespace memory {
auto allocate_aligned(std::size_t size) -> void *;

auto deallocate_aligned(void *ptr) -> void;

auto padding_for_vectorization(std::size_t type_size, std::size_t size)
    -> std::size_t;
} // namespace memory

// An array with padding of NEONUFFT_MAX_VEC_LENGTH and guaranteed alignment of
// each inner dimension
template <typename T, std::size_t DIM>
class HostArray {
public:
  static_assert(std::is_trivially_destructible_v<T>);
  static_assert(sizeof(T) <= NEONUFFT_MAX_VEC_LENGTH);
  static_assert(DIM == 1 || (NEONUFFT_MAX_VEC_LENGTH % sizeof(T)) == 0,
                "multi-dimensional array requires NEONUFFT_MAX_VEC_LENGTH "
                "be a multiple of the type size");
#ifndef NEONUFFT_ROCM  // Complex type in HIP is not marked trivially copyable
  static_assert(std::is_trivially_copyable_v<T>);
#endif

  using ValueType = T;
  using IndexType = IndexArray<DIM>;
  using SliceType = HostArray<T, DIM - 1>;

  static inline constexpr IntType dimension = DIM;

  HostArray() : data_(nullptr, &memory::deallocate_aligned) {};

  HostArray(const HostArray&) = delete;

  HostArray(HostArray&&) noexcept = default;

  auto operator=(const HostArray&) -> HostArray& = delete;

  auto operator=(HostArray&& b) noexcept -> HostArray& = default;

  operator ConstView<T, DIM>() const { return v_; };

  operator View<T, DIM>() const { return v_; };

  operator ConstHostView<T, DIM>() const { return v_; };

  operator HostView<T, DIM>() const { return v_; };

  // Allocate array of given shape, such that the inner dimension is padded for
  // alignment of each inner slice.
  HostArray(const IndexType &shape)
      : v_(nullptr, shape, shape_to_stride(shape)),
        data_(nullptr, &memory::deallocate_aligned) {
    if (v_.size()) {
      const auto inner_padding = memory::padding_for_vectorization(sizeof(T), v_.shape(0));
      auto padded_shape = shape;
      padded_shape[0] += inner_padding;
      const auto allocate_size = view_size(padded_shape);
      auto ptr =
          static_cast<T *>(memory::allocate_aligned(allocate_size * sizeof(T)));
      data_ = decltype(data_)(ptr, &memory::deallocate_aligned);

      // set to 0 to avoid issues with nan and to initialize memory page
      std::memset(ptr, 0, allocate_size * sizeof(T));
      this->v_ = HostView<T, DIM>(ptr, shape, shape_to_stride(shape));
    }
  };

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

  inline auto view() -> HostView<T, DIM> { return v_; }

  inline auto view() const -> ConstHostView<T, DIM> { return v_; }

  // resize and set to 0
  inline auto reset(const IndexType &newShape) -> void {
    if (impl::all_equal(newShape, v_.shape())) {
      v_.zero();
    } else {
      *this = HostArray<T, DIM>(newShape);
    }
  }

  inline auto operator[](const IndexType& index) const noexcept -> const T& { return v_[index]; }

  inline auto operator[](const IndexType& index) noexcept -> T& { return v_[index]; }

  inline auto data() noexcept -> T* { return const_cast<T*>(v_.data()); }

  inline auto data() const noexcept -> const T* { return v_.data(); }

  inline auto size() const noexcept -> IntType { return v_.size(); }

  inline auto size_in_bytes() const noexcept -> IntType { return v_.size_in_bytes(); }

  inline auto is_contiguous() const noexcept -> bool { return v_.is_contiguous(); }

  inline auto shape() const noexcept -> const IndexType& { return v_.shape(); }

  inline auto shape(IntType i) const noexcept -> IntType { return v_.shape(i); }

  inline auto strides() const noexcept -> const IndexType& { return v_.strides(); }

  inline auto strides(IntType i) const noexcept -> IntType { return v_.strides(i); }

  inline auto slice_view(IntType outer_index) const -> SliceType {
    return SliceType(v_.slice_view(outer_index));
  }

  inline auto sub_view(const IndexType& offset, const IndexType& shape) const -> HostView<T, DIM> {
    return HostView(v_.sub_view(offset, shape));
  }

  inline auto zero() -> void { v_.zero(); }

private:
  static inline auto shape_to_stride(const IndexType& shape) -> IndexType {
    if constexpr (DIM == 1) {
      return 1;
    } else {
      IndexType strides;
      strides[0] = 1;
      strides[1] = shape[0] + memory::padding_for_vectorization(sizeof(T), shape[0]);
      for (IntType i = 2; i < DIM; ++i) {
        strides[i] = shape[i - 1] * strides[i - 1];
      }
      return strides;
    }
  }


  HostView<T, DIM> v_;
  std::unique_ptr<T, decltype(&memory::deallocate_aligned)> data_;
};

}  // namespace neonufft
