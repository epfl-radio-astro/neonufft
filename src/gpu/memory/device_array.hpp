#pragma once

#include "neonufft/config.h"
// ---

#include <cstddef>
#include <cstring>
#include <memory>

#include "gpu/memory/device_view.hpp"
#include "neonufft/allocator.hpp"

/*
 * Arrays are views, that own the attached memory.
 */

namespace neonufft {

namespace gpu {

// An array with padding of NEONUFFT_MAX_VEC_LENGTH and guaranteed alignment of
// each inner dimension
template <typename T, std::size_t DIM>
class DeviceArray {
public:
  static_assert(std::is_trivially_destructible_v<T>);
#ifndef NEONUFFT_ROCM  // Complex type in HIP is not marked trivially copyable
  static_assert(std::is_trivially_copyable_v<T>);
#endif

  using ValueType = T;
  using IndexType = IndexArray<DIM>;
  using SliceType = DeviceArray<T, DIM - 1>;

  static inline constexpr IntType dimension = DIM;

  DeviceArray() : data_(nullptr, AllocatorWrapper()) {};

  DeviceArray(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&&) noexcept = default;

  auto operator=(const DeviceArray&) -> DeviceArray& = delete;

  auto operator=(DeviceArray&& b) noexcept -> DeviceArray& = default;

  operator ConstView<T, DIM>() const { return v_; };

  operator View<T, DIM>() const { return v_; };

  operator ConstDeviceView<T, DIM>() const { return v_; };

  operator DeviceView<T, DIM>() const { return v_; };

  // Allocate array of given shape, such that the inner dimension is padded for
  // alignment of each inner slice.
  DeviceArray(const IndexType &shape, std::shared_ptr<Allocator> alloc)
  {
    const auto allocate_size = view_size(shape);
    auto ptr = static_cast<T*>(alloc->allocate(allocate_size * sizeof(T)));

    data_ = std::unique_ptr<T, AllocatorWrapper>(ptr, AllocatorWrapper{std::move(alloc)});
    v_ = DeviceView<T, DIM>(ptr, shape, shape_to_stride(shape));
  };

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

  inline auto view() -> DeviceView<T, DIM> { return v_; }

  inline auto view() const -> ConstDeviceView<T, DIM> { return v_; }

  // resize if required. Does not copy or zero memory.
  inline auto reset(const IndexType& newShape, std::shared_ptr<Allocator> alloc) -> void {
    if (!impl::all_equal(newShape, v_.shape())) {
      *this = DeviceArray<T, DIM>(newShape, alloc);
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

  inline auto sub_view(const IndexType& offset, const IndexType& shape) const -> DeviceView<T, DIM> {
    return DeviceView(v_.sub_view(offset, shape));
  }

  inline auto zero() -> void { v_.zero(); }

private:
  static inline auto shape_to_stride(const IndexType& shape) -> IndexType {
    if constexpr (DIM == 1) {
      return 1;
    } else {
      IndexType strides;
      strides[0] = 1;
      strides[1] = shape[0];
      for (IntType i = 2; i < DIM; ++i) {
        strides[i] = shape[i - 1] * strides[i - 1];
      }
      return strides;
    }
  }

  struct AllocatorWrapper {
    std::shared_ptr<Allocator> alloc;

    auto operator()(T* ptr) noexcept-> void{
      if (alloc && ptr) alloc->deallocate(ptr);
    }

  };


  DeviceView<T, DIM> v_;
  std::unique_ptr<T, AllocatorWrapper> data_;
};

}

} // namespace neonufft
