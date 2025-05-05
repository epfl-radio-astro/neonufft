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
#include "neonufft/util/index_array.hpp"
#include "neonufft/util/func_attributes.hpp"

/*
 *
 *  Views are non-owning objects allowing access to memory through multi-dimensional indexing.
 *  Arrays behave like views and own the associated memory.
 *
 *  Note: Coloumn-major memory layout! The stride in the first dimension is always 1.
 *
 *  The type conversion tree is as follows:
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
template <IntType DIM, IntType N>
struct ViewIndexHelper {
  NEONUFFT_H_D_FUNC inline static constexpr auto eval(const IndexArray<DIM>& indices,
                                                      const IndexArray<DIM>& strides) -> IntType {
    return indices[N] * strides[N] + ViewIndexHelper<DIM, N - 1>::eval(indices, strides);
  }
};

template <IntType DIM>
struct ViewIndexHelper<DIM, 0> {
  NEONUFFT_H_D_FUNC inline static constexpr auto eval(const IndexArray<DIM>& indices,
                                                      const IndexArray<DIM>&) -> IntType {
    static_assert(DIM >= 1);
    return indices[0];
  }
};

template <IntType DIM>
struct ViewIndexHelper<DIM, 1> {
  NEONUFFT_H_D_FUNC inline static constexpr auto eval(const IndexArray<DIM>& indices,
                                                      const IndexArray<DIM>& strides) -> IntType {
    static_assert(DIM >= 2);
    return indices[0] + indices[1] * strides[1];
  }
};

template <IntType DIM>
NEONUFFT_H_D_FUNC inline auto all_less(const IndexArray<DIM>& left, const IndexArray<DIM>& right)
    -> bool {
  bool res = left[0] < right[0];
  for (IntType i = 1; i < DIM; ++i) {
    res &= left[i] < right[i];
  }

  return res;
}

template <IntType DIM>
NEONUFFT_H_D_FUNC inline auto all_equal(const IndexArray<DIM>& left, const IndexArray<DIM>& right)
    -> bool {
  bool res = left[0] == right[0];
  for (IntType i = 1; i < DIM; ++i) {
    res &= left[i] == right[i];
  }

  return res;
}

}  // namespace impl

/*
 * Helper functions
 */
template <IntType DIM>
NEONUFFT_H_D_FUNC inline constexpr auto view_index(const IndexArray<DIM>& indices,
                                                   const IndexArray<DIM>& strides) -> IntType {
  return impl::ViewIndexHelper<DIM, DIM - 1>::eval(indices, strides);
}

template <IntType DIM>
NEONUFFT_H_D_FUNC inline constexpr auto view_size(const IndexArray<DIM>& shape) -> IntType {
  if constexpr (DIM == 0) {
    return 0;
  }
  IntType size = 1;
  for (IntType i = 0; i < DIM; ++i) {
    size *= shape[i];
  }
  return size;
}

template <typename T, IntType DIM>
class ConstView {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = IndexArray<DIM>;
  using SliceType = ConstView<T, DIM - 1>;

  static inline constexpr IntType dimension = DIM;

  NEONUFFT_H_D_FUNC ConstView() {
    shape_.fill(0);
    strides_.fill(1);
  }

  NEONUFFT_H_D_FUNC ConstView(const ConstView&) = default;

  NEONUFFT_H_D_FUNC ConstView(ConstView&&) = default;

  NEONUFFT_H_D_FUNC ConstView& operator=(const ConstView&) = default;

  NEONUFFT_H_D_FUNC ConstView& operator=(ConstView&&) = default;

  NEONUFFT_H_D_FUNC ConstView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : shape_(shape), strides_(strides), totalSize_(view_size(shape)), constPtr_(ptr) {
#ifndef NDEBUG
    assert(this->strides(0) == 1);
    for (IntType i = 1; i < DIM; ++i) {
      assert(this->strides(i) >= this->shape(i - 1) * this->strides(i - 1));
    }
#endif
  }

  NEONUFFT_H_D_FUNC inline auto data() const noexcept -> const T* { return constPtr_; }

  NEONUFFT_H_D_FUNC inline auto size() const noexcept -> IntType { return totalSize_; }

  NEONUFFT_H_D_FUNC inline auto size_in_bytes() const noexcept -> IntType {
    return totalSize_ * sizeof(T);
  }

  NEONUFFT_H_D_FUNC inline auto is_contiguous() const noexcept -> bool {
    if constexpr (DIM <= 1) {
      return true;
    }

    bool res = true;
    for (IntType i = 0; i < shape_.size() - 1; ++i) {
      res &= strides_[i + 1] == shape_[i] * strides_[i];
    }
    return res;
  }

  NEONUFFT_H_D_FUNC inline auto shape() const noexcept -> const IndexType& { return shape_; }

  NEONUFFT_H_D_FUNC inline auto shape(IntType i) const noexcept -> IntType {
    assert(i < DIM);
    return shape_[i];
  }

  NEONUFFT_H_D_FUNC inline auto strides() const noexcept -> const IndexType& { return strides_; }

  NEONUFFT_H_D_FUNC inline auto strides(IntType i) const noexcept -> IntType {
    assert(i < DIM);
    return strides_[i];
  }

  NEONUFFT_H_D_FUNC auto slice_view(IntType outer_index) const -> SliceType {
    typename SliceType::IndexType slice_shape, slice_strides;

    for (IntType i = 0; i < slice_shape.size(); ++i) {
      slice_shape[i] = shape_[i];
      slice_strides[i] = strides_[i];
    }

    return SliceType{this->constPtr_ + outer_index * this->strides(DIM - 1), slice_shape,
                     slice_strides};
  }

  NEONUFFT_H_D_FUNC auto sub_view(const IndexType& offset, const IndexType& shape) const
      -> ConstView<T, DIM> {
    assert(impl::all_less(offset, shape_));
#ifndef NDEBUG
    for (IntType i = 0; i < DIM; ++i) {
      assert(shape[i] + offset[i] <= shape_[i]);
    }
#endif

    return ConstView{constPtr_ + view_index(offset, strides_), shape, strides_};
  }

private:
  friend ConstView<T, DIM + 1>;

  IndexType shape_;
  IndexType strides_;
  IntType totalSize_ = 0;
  const T* constPtr_ = nullptr;
};

template <typename T, IntType DIM>
class View {
public:
  using ValueType = T;
  using IndexType = IndexArray<DIM>;
  using SliceType = View<T, DIM - 1>;

  static inline constexpr IntType dimension = DIM;

  NEONUFFT_H_D_FUNC View() = default;

  NEONUFFT_H_D_FUNC View(const View&) = default;

  NEONUFFT_H_D_FUNC View(View&&) = default;

  NEONUFFT_H_D_FUNC View& operator=(const View&) = default;

  NEONUFFT_H_D_FUNC View& operator=(View&&) = default;

  NEONUFFT_H_D_FUNC View(T* ptr, const IndexType& shape, const IndexType& strides)
      : v_(ptr, shape, strides) {}

  NEONUFFT_H_D_FUNC operator ConstView<T, DIM>() const { return v_; };

  NEONUFFT_H_D_FUNC inline auto data() noexcept -> T* { return const_cast<T*>(v_.data()); }

  NEONUFFT_H_D_FUNC inline auto data() const noexcept -> const T* { return v_.data(); }

  NEONUFFT_H_D_FUNC inline auto size() const noexcept -> IntType { return v_.size(); }

  NEONUFFT_H_D_FUNC inline auto size_in_bytes() const noexcept -> IntType {
    return v_.size_in_bytes();
  }

  NEONUFFT_H_D_FUNC inline auto is_contiguous() const noexcept -> bool {
    return v_.is_contiguous();
  }

  NEONUFFT_H_D_FUNC inline auto shape() const noexcept -> const IndexType& { return v_.shape(); }

  NEONUFFT_H_D_FUNC inline auto shape(IntType i) const noexcept -> IntType { return v_.shape(i); }

  NEONUFFT_H_D_FUNC inline auto strides() const noexcept -> const IndexType& {
    return v_.strides();
  }

  NEONUFFT_H_D_FUNC inline auto strides(IntType i) const noexcept -> IntType {
    return v_.strides(i);
  }

  NEONUFFT_H_D_FUNC inline auto slice_view(IntType outer_index) const -> SliceType {
    return v_.slice_view(outer_index);
  }

  NEONUFFT_H_D_FUNC inline auto sub_view(const IndexType& offset, const IndexType& shape) const
      -> View<T, DIM> {
    return View(v_.sub_view(offset, shape));
  }

private:
  friend View<T, DIM + 1>;

  NEONUFFT_H_D_FUNC View(const ConstView<T, DIM>& v) : v_(v) {}

  ConstView<T, DIM> v_;
};

template <typename T, IntType DIM>
class HostView {
public:
  using ValueType = T;
  using IndexType = IndexArray<DIM>;
  using SliceType = HostView<T, DIM - 1>;

  static inline constexpr IntType dimension = DIM;

  HostView() = default;

  explicit HostView(const View<T, DIM>& v) : v_(v) {};

  HostView(T* ptr, const IndexType& shape, const IndexType& strides) : v_(ptr, shape, strides) {};

  operator ConstView<T, DIM>() const { return v_; };

  operator View<T, DIM>() const { return v_; };

  inline auto operator[](const IndexType& index) const noexcept -> const T& {
    assert(impl::all_less(index, v_.shape()));
    return v_.data()[view_index(index, v_.strides())];
  }

  inline auto operator[](const IndexType& index) noexcept -> T& {
    assert(impl::all_less(index, v_.shape()));
    return v_.data()[view_index(index, v_.strides())];
  }

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

  inline auto zero() -> void {
    if (v_.size()) {
      if constexpr (DIM <= 1) {
        std::memset(v_.data(), 0, v_.shape(0) * sizeof(T));
      } else {
        for (IntType i = 0; i < v_.shape(DIM - 1); ++i) this->slice_view(i).zero();
      }
    }
  }

private:
  View<T, DIM> v_;
};

template <typename T, IntType DIM>
class ConstHostView {
public:
  using ValueType = T;
  using IndexType = IndexArray<DIM>;
  using SliceType = ConstHostView<T, DIM - 1>;

  static inline constexpr IntType dimension = DIM;

  ConstHostView() = default;

  explicit ConstHostView(const ConstView<T, DIM>& v) : v_(v) {};

  ConstHostView(const HostView<T, DIM>& v) : v_(v) {};

  ConstHostView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : v_(ptr, shape, strides) {};

  operator ConstView<T, DIM>() const { return v_; };

  inline auto operator[](const IndexType& index) const noexcept -> const T& {
    assert(impl::all_less(index, v_.shape()));
    return v_.data()[view_index(index, v_.strides())];
  }

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

  inline auto sub_view(const IndexType& offset, const IndexType& shape) const
      -> ConstHostView<T, DIM> {
    return ConstHostView(v_.sub_view(offset, shape));
  }

private:
  ConstView<T, DIM> v_;
};
}  // namespace neonufft
