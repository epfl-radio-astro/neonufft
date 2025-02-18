#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>

#include "neonufft/config.h"
#include "neonufft/exceptions.hpp"
#include "neonufft/types.hpp"
#include "memory/view.hpp"

/*
 *
 *  Views are non-owning objects allowing access to memory through multi-dimensional indexing.
 *  Arrays inherit from views and own the associated memory.
 *
 *  Note: Coloumn-major memory layout! The stride in the first dimension is always 1.
 *
 *  The inheritance tree is as follows:
 *
 *                            ConstView
 *                                |
 *                ------------------------View
 *                |                         |
 *          --------------              -----------
 *          |            |              |         |
 *  ConstHostView  ConstDeviceView  HostView DeviceView
 *                                      |         |
 *                                 HostArray DeviceArray
 *
 *
 * Host views support the [..] operator for element-wise access.
 * Device views may not be passed to device kernels and do not support element-wise access.
 */

namespace neonufft {
namespace gpu {

template <typename T, std::size_t DIM>
class DeviceView : public View<T, DIM> {
public:
  using ValueType = T;
  using BaseType = View<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = DeviceView<T, DIM - 1>;

  DeviceView() : BaseType(){};

  explicit DeviceView(const View<T, DIM>& v) : BaseType(v){};

  DeviceView(T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> DeviceView<T, DIM> {
    return this->template sub_view_impl<DeviceView<T, DIM>>(offset, shape);
  }

  inline auto shrink(const IndexType& newShape) -> DeviceView<T, DIM>& {
    this->shrink_impl(newShape);
    return *this;
  }

protected:
  friend ConstView<T, DIM>;
  friend ConstView<T, DIM + 1>;

  DeviceView(const ConstView<T, DIM>& b) : BaseType(b){};
};

template <typename T, std::size_t DIM>
class ConstDeviceView : public ConstView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = ConstDeviceView<T, DIM - 1>;

  ConstDeviceView() : BaseType(){};

  ConstDeviceView(const DeviceView<T, DIM>& v) : BaseType(v){};

  explicit ConstDeviceView(const ConstView<T, DIM>& v) : BaseType(v){};

  ConstDeviceView(std::shared_ptr<Allocator> alloc, const IndexType& shape)
      : BaseType(std::move(alloc), shape){};

  ConstDeviceView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> ConstDeviceView<T, DIM> {
    return this->template sub_view_impl<ConstDeviceView<T, DIM>>(offset, shape);
  }

  inline auto shrink(const IndexType& newShape) -> ConstDeviceView<T, DIM>& {
    this->shrink_impl(newShape);
    return *this;
  }
};

} // namespace gpu
}  // namespace neonufft
