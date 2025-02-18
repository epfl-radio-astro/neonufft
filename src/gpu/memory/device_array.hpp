#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>

#include "neonufft/config.h"

#include "gpu/memory/device_view.hpp"


/*
 * Arrays are views, that own the attached memory.
 */

namespace neonufft {

namespace gpu {

template <typename T, std::size_t DIM>
class DeviceArray : public DeviceView<T, DIM> {
public:
  static_assert(std::is_trivially_destructible_v<T>);
#ifndef NEONUFFT_ROCM
  static_assert(std::is_trivially_copyable_v<T>);
#endif

  using ValueType = T;
  using BaseType = DeviceView<T, DIM>;
  using IndexType = typename View<T, DIM>::IndexType;
  using SliceType = DeviceArray<T, DIM - 1>;

  DeviceArray() : BaseType(){};

  DeviceArray(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&&) noexcept = default;

  auto operator=(const DeviceArray&) -> DeviceArray& = delete;

  auto operator=(DeviceArray&& b) noexcept -> DeviceArray& = default;

  DeviceArray(std::shared_ptr<Allocator> alloc, const IndexType& shape)
      : BaseType(nullptr, shape, shape_to_stride(shape)) {
    if (alloc->type() != MemoryType::Device)
      throw InternalError("View: Memory type and allocator type mismatch.");
    if (this->totalSize_) {
      auto ptr = alloc->allocate(this->totalSize_ * sizeof(T));
      data_ = std::shared_ptr<void>(ptr, [alloc = std::move(alloc)](void* p) {
        if (p) alloc->deallocate(p);
      });
      this->constPtr_ = static_cast<T*>(ptr);
    }
  };

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

  inline auto view() -> DeviceView<T, DIM> { return *this; }

  inline auto view() const -> ConstDeviceView<T, DIM> { return *this; }

  inline auto shrink(const IndexType& newShape) -> DeviceArray<T, DIM>& {
    this->shrink_impl(newShape);
    return *this;
  }

private:
  static inline auto shape_to_stride(const IndexType& shape) -> IndexType {
    if constexpr (DIM == 1) {
      return 1;
    } else {
      IndexType strides;
      strides[0] = 1;
      for (std::size_t i = 1; i < DIM; ++i) {
        strides[i] = shape[i - 1] * strides[i - 1];
      }
      strides[0] = 1;
      return strides;
    }
  }
  std::shared_ptr<void> data_;
};

}

} // namespace neonufft
