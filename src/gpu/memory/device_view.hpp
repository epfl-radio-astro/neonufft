#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>

#include "gpu/util//runtime_api.hpp"
#include "memory/view.hpp"
#include "neonufft/config.h"
#include "neonufft/exceptions.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/types.hpp"
#include "util/func_attributes.hpp"

/*
 *
 *  Views are non-owning objects allowing access to memory through multi-dimensional indexing.
 *  Arrays behave like views and own the associated memory.
 *
 *  Note: Coloumn-major memory layout! The stride in the first dimension is always 1.
 *
 *  The conversion tree is as follows:
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

template <typename T, IntType DIM>
class DeviceView {
public:
  using ValueType = T;
  using IndexType = IndexArray<DIM>;
  using SliceType = DeviceView<T, DIM - 1>;

  NEONUFFT_H_D_FUNC DeviceView() = default;

  NEONUFFT_H_D_FUNC explicit DeviceView(const View<T, DIM>& v) : v_(v){};

  NEONUFFT_H_D_FUNC DeviceView(T* ptr, const IndexType& shape, const IndexType& strides)
      : v_(ptr, shape, strides) {};

  NEONUFFT_H_D_FUNC operator ConstView<T, DIM>() const { return v_; };

  NEONUFFT_H_D_FUNC operator View<T, DIM>() const { return v_; };

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __forceinline__ inline auto operator[](const IndexType& index) const noexcept
      -> const T& {
    assert(impl::all_less(index, v_.shape()));
    return v_.data()[view_index(index, v_.strides())];
  }

  __device__ __forceinline__ inline auto operator[](const IndexType& index) noexcept -> T& {
    assert(impl::all_less(index, v_.shape()));
    return v_.data()[view_index(index, v_.strides())];
  }
#endif

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
    return SliceType(v_.slice_view(outer_index));
  }

  NEONUFFT_H_D_FUNC inline auto sub_view(const IndexType& offset, const IndexType& shape) const
      -> DeviceView<T, DIM> {
    return DeviceView(v_.sub_view(offset, shape));
  }

  NEONUFFT_H_FUNC inline auto zero(const StreamType& stream) -> void {
    if (v_.size()) {
      if constexpr (DIM == 1) {
        // break function call recursion
        api::memset_async(v_.data(), 0, v_.shape(0) * sizeof(T), stream);
      } else {
        if(v_.is_contiguous() ) {
          api::memset_async(v_.data(), 0, v_.size(0) * sizeof(T), stream);
        } else {
          if constexpr (DIM == 2) {
            api::memset_2d_async(v_.data(), v_.strides(1), 0, v_.shape(0) * sizeof(T), v_.shape(1),
                                 stream);
          } else {
            for (IntType i = 0; i < v_.shape(DIM - 1); ++i) this->slice_view(i).zero();
          }
        }
      }
    }
  }

private:
  View<T, DIM> v_;
};

template <typename T, IntType DIM>
class ConstDeviceView {
public:
  using ValueType = T;
  using IndexType = IndexArray<DIM>;
  using SliceType = ConstDeviceView<T, DIM - 1>;

  NEONUFFT_H_D_FUNC ConstDeviceView() = default;

  NEONUFFT_H_D_FUNC explicit ConstDeviceView(const ConstView<T, DIM>& v) : v_(v){};

  NEONUFFT_H_D_FUNC ConstDeviceView(const DeviceView<T, DIM>& v) : v_(v) {};

  NEONUFFT_H_D_FUNC ConstDeviceView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : v_(ptr, shape, strides) {};

  NEONUFFT_H_D_FUNC operator ConstView<T, DIM>() const { return v_; };

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __forceinline__ inline auto operator[](const IndexType& index) const noexcept -> T {
    assert(impl::all_less(index, v_.shape()));
    return __ldg(v_.data() + view_index(index, v_.strides()));
  }
#endif

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
    return SliceType(v_.slice_view(outer_index));
  }

  NEONUFFT_H_D_FUNC inline auto sub_view(const IndexType& offset, const IndexType& shape) const
      -> ConstDeviceView<T, DIM> {
    return ConstDeviceView(v_.sub_view(offset, shape));
  }

private:
  ConstView<T, DIM> v_;
};

} // namespace gpu
}  // namespace neonufft
