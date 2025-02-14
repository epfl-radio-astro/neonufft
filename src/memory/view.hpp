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

namespace impl {
// Use specialized structs to compute index, since some compiler do not properly optimize otherwise
template <std::size_t DIM, IntType N>
struct ViewIndexHelper {
  inline static constexpr auto eval(const std::array<IntType, DIM>& indices,
                                    const std::array<IntType, DIM>& strides) -> IntType {
    return indices[N] * strides[N] + ViewIndexHelper<DIM, N - 1>::eval(indices, strides);
  }
};

template <std::size_t DIM>
struct ViewIndexHelper<DIM, 0> {
  inline static constexpr auto eval(const std::array<IntType, DIM>& indices,
                                    const std::array<IntType, DIM>&) -> IntType {
    static_assert(DIM >= 1);
    return indices[0];
  }
};

template <std::size_t DIM>
struct ViewIndexHelper<DIM, 1> {
  inline static constexpr auto eval(const std::array<IntType, DIM>& indices,
                                    const std::array<IntType, DIM>& strides) -> IntType {
    static_assert(DIM >= 2);
    return indices[0] + indices[1] * strides[1];
  }
};

}  // namespace impl

/*
 * Helper functions
 */
template <std::size_t DIM>
inline constexpr auto view_index(const std::array<IntType, DIM>& indices,
                                  const std::array<IntType, DIM>& strides) -> IntType {
  return impl::ViewIndexHelper<DIM, DIM - 1>::eval(indices, strides);
}

inline constexpr auto view_index(IntType index, IntType) -> IntType { return index; }

inline constexpr auto view_size(IntType shape) -> IntType { return shape; }

template <std::size_t DIM>
inline constexpr auto view_size(const std::array<IntType, DIM>& shape) -> IntType {
  return std::reduce(shape.begin(), shape.end(), IntType(1), std::multiplies{});
}



template <typename T, std::size_t DIM>
class ConstView {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = std::conditional_t<DIM == 1, IntType, std::array<IntType, DIM>>;
  using SliceType = ConstView<T, DIM - 1>;

  ConstView() {
    if constexpr(DIM==1) {
      shape_ = 0;
      strides_ = 1;
    } else {
      shape_.fill(0);
      strides_.fill(1);
    }
  }

  ConstView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : shape_(shape), strides_(strides), totalSize_(view_size(shape)), constPtr_(ptr) {
#ifndef NDEBUG
    assert(this->strides(0) == 1);
    for (IntType i = 1; i < DIM; ++i) {
      assert(this->strides(i) >= this->shape(i - 1) * this->strides(i - 1));
    }
#endif
  }

  virtual ~ConstView() = default;

  inline auto data() const -> const T* { return constPtr_; }

  inline auto size() const noexcept -> IntType { return totalSize_; }

  inline auto size_in_bytes() const noexcept -> IntType { return totalSize_ * sizeof(T); }

  inline auto is_contiguous() const noexcept -> bool {
    if constexpr (DIM == 1) {
      return true;
    }
    return std::transform_reduce(shape_.begin(), shape_.end() - 1, strides_.begin() + 1, true,
                                 std::logical_and{}, std::equal_to{});
  }

  inline auto shape() const noexcept -> const IndexType& { return shape_; }

  inline auto shape(IntType i) const noexcept -> IntType {
    assert(i < DIM);
    if constexpr (DIM == 1)
      return shape_;
    else
      return shape_[i];
  }

  inline auto strides() const noexcept -> const IndexType& { return strides_; }

  inline auto strides(IntType i) const noexcept -> IntType {
    assert(i < DIM);
    if constexpr (DIM == 1)
      return strides_;
    else
      return strides_[i];
  }

  template <typename F>
  auto cast_to_type() -> ConstView<T, DIM> {
    static_assert(sizeof(F) == sizeof(T));
    return ConstView<F, DIM>(reinterpret_cast<const F*>(constPtr_), shape_, strides_);
  };

  auto slice_view(IntType outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> ConstView<T, DIM> {
    return this->template sub_view_impl<ConstView<T, DIM>>(offset, shape);
  }

  inline auto shrink(const IndexType& newShape) -> ConstView<T, DIM>& {
    this->shrink_impl(newShape);
    return *this;
  }

protected:
  friend ConstView<T, DIM + 1>;

  template <typename UnaryTransformOp>
  inline auto compare_elements(const IndexType& left, const IndexType& right,
                               UnaryTransformOp&& op) const -> bool {
    if constexpr (DIM == 1) {
      return op(left, right);
    } else {
      return std::transform_reduce(left.begin(), left.end(), right.begin(), true,
                                   std::logical_and{}, std::forward<UnaryTransformOp>(op));
    }
  }

  template <typename SLICE_TYPE>
  auto slice_view_impl(IntType outer_index) const -> SLICE_TYPE {
    assert(outer_index < this->shape(DIM - 1));

    typename SLICE_TYPE::IndexType sliceShape, sliceStrides;
    if constexpr(DIM == 2) {
      sliceShape = shape_[0];
      sliceStrides = strides_[0];
    } else {
      std::copy(this->shape_.begin(), this->shape_.end() - 1, sliceShape.begin());
      std::copy(this->strides_.begin(), this->strides_.end() - 1, sliceStrides.begin());
    }

    return SLICE_TYPE{ConstView<T, DIM - 1>{this->constPtr_ + outer_index * this->strides(DIM - 1),
                                           sliceShape, sliceStrides}};
  }

  template <typename VIEW_TYPE>
  auto sub_view_impl(const IndexType& offset, const IndexType& shape) const -> VIEW_TYPE {
    assert(compare_elements(offset, shape_, std::less{}));
#ifndef NDEBUG
    for (IntType i = 0; i < DIM; ++i) {
      if constexpr (DIM == 1) {
        assert(shape + offset <= shape_);
      } else {
        assert(shape[i] + offset[i] <= shape_[i]);
      }
    }
#endif

    return VIEW_TYPE{ConstView{constPtr_ + view_index(offset, strides_), shape, strides_}};
  }

  inline auto shrink_impl(const IndexType& newShape) -> void {
    assert(compare_elements(newShape, shape_, std::less_equal{}));

    totalSize_ = view_size(newShape);
    shape_ = newShape;
  }

  IndexType shape_;
  IndexType strides_;
  IntType totalSize_ = 0;
  const T* constPtr_ = nullptr;
};

template <typename T, std::size_t DIM>
class View : public ConstView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = View<T, DIM - 1>;

  View() : BaseType() {}

  View(T* ptr, const IndexType& shape, const IndexType& strides) : BaseType(ptr, shape, strides) {}

  inline auto data() -> T* { return const_cast<T*>(this->constPtr_); }

  template <typename F>
  auto cast_to_type() -> View<T, DIM> {
    static_assert(sizeof(F) == sizeof(T));
    return View<F, DIM>(reinterpret_cast<const F*>(this->constPtr_), this->shape_, this->strides_);
  };

  auto slice_view(IntType outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> View<T, DIM> {
    return this->template sub_view_impl<View<T, DIM>>(offset, shape);
  }

  inline auto shrink(const IndexType& newShape) -> View<T, DIM>& {
    this->shrink_impl(newShape);
    return *this;
  }

protected:
  friend ConstView<T, DIM>;
  friend ConstView<T, DIM + 1>;

  View(const BaseType& v) : BaseType(v) {}
};

template <typename T, std::size_t DIM>
class HostView : public View<T, DIM> {
public:
  using ValueType = T;
  using BaseType = View<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = HostView<T, DIM - 1>;

  HostView() : BaseType(){};

  explicit HostView(const View<T, DIM>& v) : BaseType(v){};

  HostView(T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  inline auto operator[](const IndexType& index) const -> const T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return this->constPtr_[view_index(index, this->strides_)];
  }

  inline auto operator[](const IndexType& index) -> T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return const_cast<T*>(this->constPtr_)[view_index(index, this->strides_)];
  }

  auto slice_view(IntType outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> HostView<T, DIM> {
    return this->template sub_view_impl<HostView<T, DIM>>(offset, shape);
  }

  auto zero() -> void {
    if (this->totalSize_) {
      if constexpr (DIM <= 1) {
        std::memset(this->data(), 0, this->shape_ * sizeof(T));
      } else {
        for (IntType i = 0; i < this->shape_[DIM - 1]; ++i) this->slice_view(i).zero();
      }
    }
  }

  inline auto shrink(const IndexType& newShape) -> HostView<T, DIM>& {
    this->shrink_impl(newShape);
    return *this;
  }

protected:
  friend ConstView<T, DIM>;
  friend ConstView<T, DIM + 1>;

  HostView(const ConstView<T, DIM>& b) : BaseType(b){};
};


template <typename T, std::size_t DIM>
class ConstHostView : public ConstView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = ConstHostView<T, DIM - 1>;

  ConstHostView() : BaseType(){};

  ConstHostView(const HostView<T, DIM>& v) : BaseType(v){};

  explicit ConstHostView(const ConstView<T, DIM>& v) : BaseType(v){};

  ConstHostView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  inline auto operator[](const IndexType& index) const -> const T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return this->constPtr_[view_index(index, this->strides_)];
  }

  auto slice_view(IntType outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> ConstHostView<T, DIM> {
    return this->template sub_view_impl<ConstHostView<T, DIM>>(offset, shape);
  }

  inline auto shrink(const IndexType& newShape) -> ConstHostView<T, DIM>& {
    this->shrink_impl(newShape);
    return *this;
  }
};
}  // namespace neonufft
