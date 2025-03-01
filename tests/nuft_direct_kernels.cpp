#include <array>
#include <cassert>
#include <complex>
#include <cstring>

#include "neonufft/config.h"
#include "neonufft/enums.h"

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/types.hpp"
#include "nuft_direct_kernels.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "nuft_direct_kernels.cpp" // this file

#include "neonufft/kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace test {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR std::complex<T> type_2_inner(IntType k_offset, IntType mode, const std::complex<T>* in,
                                      T point, T dot_init) {
  const ::neonufft::HWY_NAMESPACE::TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);
  const T* in_scalar = reinterpret_cast<const T*>(in);
  std::complex<T> sum = {0, 0};
  auto v_sum_real = hn::Zero(d);
  auto v_sum_imag = hn::Zero(d);

  auto v_point = hn::Set(d, point);
  auto v_dot_init = hn::Set(d, dot_init);

  IntType k0 = 0;
  for (; k0 + n_lanes <= mode; k0 += n_lanes) {
    auto v_in_1 = hn::LoadU(d, in_scalar + 2 * k0);
    auto v_in_2 = hn::LoadU(d, in_scalar + 2 * k0 + n_lanes);

    auto v_in_real = hn::ConcatEven(d, v_in_2, v_in_1);
    auto v_in_imag = hn::ConcatOdd(d, v_in_2, v_in_1);
    auto v_k = hn::Iota(d, k0 - k_offset);
    auto v_dot = hn::Add(hn::Mul(v_point, v_k), v_dot_init);

    auto v_sin = hn::Undefined(d);
    auto v_cos = hn::Undefined(d);

    hn::SinCos(d, v_dot, v_sin, v_cos);

    // complex multiplication
    auto res_real = hn::Sub(hn::Mul(v_in_real, v_cos), hn::Mul(v_in_imag, v_sin));
    auto res_imag = hn::Add(hn::Mul(v_in_real, v_sin), hn::Mul(v_in_imag, v_cos));

    v_sum_real = hn::Add(v_sum_real, res_real);
    v_sum_imag = hn::Add(v_sum_imag, res_imag);
  }
  sum += std::complex<T>{hn::ReduceSum(d, v_sum_real), hn::ReduceSum(d, v_sum_imag)};

  for (; k0 < mode; ++k0) {
    T dot = (k0 - k_offset) * point + dot_init;

    sum += in[k0] * std::exp(std::complex<T>{0, dot});
  }

  return sum;
}

template <typename T>
HWY_ATTR void type_2_d1(int sign, std::array<IntType, 1> modes, const std::complex<T>* in,
                        IntType num_out, std::array<const T*, 1> out_points, std::complex<T>* out,
                        std::array<IntType, 1>) {
  const IntType k_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    out[idx_out] = type_2_inner<T>(k_offset, modes[0], in, sign * out_points[0][idx_out], 0);
  }
}

template <typename T>
HWY_ATTR void type_2_d2(int sign, std::array<IntType, 2> modes, const std::complex<T>* in,
                        IntType num_out, std::array<const T*, 2> out_points, std::complex<T>* out,
                        std::array<IntType, 2> strides) {
  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    std::complex<T> sum = {0, 0};

    for (IntType k1 = 0; k1 < modes[1]; ++k1) {
      const T dot = sign * (k1 - k1_offset) * out_points[1][idx_out];
      sum += type_2_inner<T>(k0_offset, modes[0], in + k1 * strides[1],
                             sign * out_points[0][idx_out], dot);
    }
    out[idx_out] = sum;
  }
}

template <typename T>
HWY_ATTR void type_2_d3(int sign, std::array<IntType, 3> modes, const std::complex<T>* in,
                        IntType num_out, std::array<const T*, 3> out_points, std::complex<T>* out,
                        std::array<IntType, 3> strides) {
  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;
  const IntType k2_offset = modes[2] % 2 ? (modes[2] - 1) / 2 : modes[2] / 2;

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    std::complex<T> sum = {0, 0};

    for (IntType k2 = 0; k2 < modes[2]; ++k2) {
      for (IntType k1 = 0; k1 < modes[1]; ++k1) {
        const T dot = sign * ((k2 - k2_offset) * out_points[2][idx_out] +
                              (k1 - k1_offset) * out_points[1][idx_out]);
        sum += type_2_inner<T>(k0_offset, modes[0], in + k1 * strides[1] + k2 * strides[2],
                               sign * out_points[0][idx_out], dot);
      }
    }
    out[idx_out] = sum;
  }
}

template <typename T, IntType DIM>
HWY_ATTR std::complex<T> type_1_inner(std::array<IntType, DIM> k, IntType num_in,
                                      const std::complex<T>* in, std::array<const T*, DIM> points) {
  const ::neonufft::HWY_NAMESPACE::TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);
  const T* in_scalar = reinterpret_cast<const T*>(in);
  std::complex<T> sum = {0, 0};
  auto v_sum_real = hn::Zero(d);
  auto v_sum_imag = hn::Zero(d);


  IntType idx_in = 0;
  for (; idx_in + n_lanes <= num_in; idx_in += n_lanes) {
    auto v_in_1 = hn::LoadU(d, in_scalar + 2 * idx_in);
    auto v_in_2 = hn::LoadU(d, in_scalar + 2 * idx_in + n_lanes);

    auto v_in_real = hn::ConcatEven(d, v_in_2, v_in_1);
    auto v_in_imag = hn::ConcatOdd(d, v_in_2, v_in_1);

    auto v_dot = hn::Mul(hn::LoadU(d, points[0] + idx_in), hn::Set(d, k[0]));
    if constexpr (DIM > 1) {
      v_dot = hn::Add(v_dot, hn::Mul(hn::LoadU(d, points[1] + idx_in), hn::Set(d, k[1])));
    }
    if constexpr (DIM > 2) {
      v_dot = hn::Add(v_dot, hn::Mul(hn::LoadU(d, points[2] + idx_in), hn::Set(d, k[2])));
    }

    auto v_sin = hn::Undefined(d);
    auto v_cos = hn::Undefined(d);

    hn::SinCos(d, v_dot, v_sin, v_cos);

    // complex multiplication
    auto res_real = hn::Sub(hn::Mul(v_in_real, v_cos), hn::Mul(v_in_imag, v_sin));
    auto res_imag = hn::Add(hn::Mul(v_in_real, v_sin), hn::Mul(v_in_imag, v_cos));

    v_sum_real = hn::Add(v_sum_real, res_real);
    v_sum_imag = hn::Add(v_sum_imag, res_imag);
  }
  sum += std::complex<T>{hn::ReduceSum(d, v_sum_real), hn::ReduceSum(d, v_sum_imag)};

  for (; idx_in < num_in; ++idx_in) {
    T dot = k[0] * points[0][idx_in];
    if constexpr (DIM > 1) {
      dot += k[1] * points[1][idx_in];
    }
    if constexpr (DIM > 2) {
      dot += k[2] * points[2][idx_in];
    }

    sum += in[idx_in] * std::exp(std::complex<T>{0, dot});
  }

  return sum;
}

template <typename T>
HWY_ATTR void type_1_d1(int sign, std::array<IntType, 1> modes, const std::complex<T>* in, IntType num_in,
               std::array<const T*, 1> in_points, std::complex<T>* out, std::array<IntType, 1>) {
  const IntType k_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;

  for (IntType k0 = 0; k0 < modes[0]; ++k0) {
    out[k0] = type_1_inner<T, 1>({(k0 - k_offset) * sign}, num_in, in, in_points);
  }
}

template <typename T>
HWY_ATTR void type_1_d2(int sign, std::array<IntType, 2> modes, const std::complex<T>* in,
                        IntType num_in, std::array<const T*, 2> in_points, std::complex<T>* out,
                        std::array<IntType, 2> strides) {
  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;

  for (IntType k1 = 0; k1 < modes[1]; ++k1) {
    for (IntType k0 = 0; k0 < modes[0]; ++k0) {
      out[k0 + k1 * strides[1]] = type_1_inner<T, 2>(
          {(k0 - k0_offset) * sign, (k1 - k1_offset) * sign}, num_in, in, in_points);
    }
  }
}

template <typename T>
HWY_ATTR void type_1_d3(int sign, std::array<IntType, 3> modes, const std::complex<T>* in,
                        IntType num_in, std::array<const T*, 3> in_points, std::complex<T>* out,
                        std::array<IntType, 3> strides) {
  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;
  const IntType k2_offset = modes[2] % 2 ? (modes[2] - 1) / 2 : modes[2] / 2;

  for (IntType k2 = 0; k2 < modes[2]; ++k2) {
    for (IntType k1 = 0; k1 < modes[1]; ++k1) {
      for (IntType k0 = 0; k0 < modes[0]; ++k0) {
        out[k0 + k1 * strides[1] + k2 * strides[2]] = type_1_inner<T, 3>(
            {(k0 - k0_offset) * sign, (k1 - k1_offset) * sign, (k2 - k2_offset) * sign}, num_in, in,
            in_points);
      }
    }
  }
}

template <typename T, IntType DIM>
HWY_ATTR void type_3(int sign, IntType num_in, std::array<const T*, DIM> in_points,
                     const std::complex<T>* in, IntType num_out,
                     std::array<const T*, DIM> out_points, std::complex<T>* out) {
  const ::neonufft::HWY_NAMESPACE::TagType<T> d;
  constexpr IntType n_lanes = hn::Lanes(d);
  const T* in_scalar = reinterpret_cast<const T*>(in);

  auto v_sign = hn::Set(d, sign);

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    auto v_sum_real = hn::Zero(d);
    auto v_sum_imag = hn::Zero(d);

    auto v_point_out_x = hn::Set(d, out_points[0][idx_out]);
    auto v_point_out_y = hn::Undefined(d);
    auto v_point_out_z = hn::Undefined(d);
    if constexpr (DIM > 1) {
      v_point_out_y = hn::Set(d, out_points[1][idx_out]);
    }
    if constexpr (DIM > 2) {
      v_point_out_z = hn::Set(d, out_points[2][idx_out]);
    }

    IntType idx_in = 0;
    for (; idx_in + n_lanes <= num_in; idx_in += n_lanes) {
      auto v_in_1 = hn::LoadU(d, in_scalar + 2 * idx_in);
      auto v_in_2 = hn::LoadU(d, in_scalar + 2 * idx_in + n_lanes);

      auto v_in_real = hn::ConcatEven(d, v_in_2, v_in_1);
      auto v_in_imag = hn::ConcatOdd(d, v_in_2, v_in_1);

      auto v_dot =
          hn::Mul(hn::LoadU(d, in_points[0] + idx_in), v_point_out_x);
      if constexpr (DIM > 1) {
        v_dot = hn::Add(v_dot, hn::Mul(hn::LoadU(d, in_points[1] + idx_in), v_point_out_y));
      }
      if constexpr (DIM > 2) {
        v_dot = hn::Add(v_dot, hn::Mul(hn::LoadU(d, in_points[2] + idx_in), v_point_out_z));
      }

      v_dot = hn::Mul(v_dot, v_sign);

      auto v_sin = hn::Undefined(d);
      auto v_cos = hn::Undefined(d);

      hn::SinCos(d, v_dot, v_sin, v_cos);

      // complex multiplication
      auto res_real = hn::Sub(hn::Mul(v_in_real, v_cos), hn::Mul(v_in_imag, v_sin));
      auto res_imag = hn::Add(hn::Mul(v_in_real, v_sin), hn::Mul(v_in_imag, v_cos));

      v_sum_real = hn::Add(v_sum_real, res_real);
      v_sum_imag = hn::Add(v_sum_imag, res_imag);
    }

    std::complex<T> sum{hn::ReduceSum(d, v_sum_real), hn::ReduceSum(d, v_sum_imag)};

    for (; idx_in < num_in; ++idx_in) {
      T dot = in_points[0][idx_in] * out_points[0][idx_out];
      for (IntType dim = 1; dim < DIM; ++dim) {
        dot += in_points[dim][idx_in] * out_points[dim][idx_out];
      }
      sum += in[idx_in] * std::exp(std::complex<T>{0, sign * dot});
    }
    out[idx_out] = sum;
  }
}

template <IntType DIM>
HWY_ATTR void type_3_float(int sign, IntType num_in, std::array<const float*, DIM> in_points,
                     const std::complex<float>* in, IntType num_out,
                     std::array<const float*, DIM> out_points, std::complex<float>* out) {
  type_3<float, DIM>(sign, num_in, in_points, in, num_out, out_points, out);
}

template <IntType DIM>
HWY_ATTR void type_3_double(int sign, IntType num_in, std::array<const double*, DIM> in_points,
                     const std::complex<double>* in, IntType num_out,
                     std::array<const double*, DIM> out_points, std::complex<double>* out) {
  type_3<double, DIM>(sign, num_in, in_points, in, num_out, out_points, out);
}

}  // namespace HWY_NAMESPACE
}  // namespace

#if HWY_ONCE

template <typename T, IntType DIM>
void nuft_direct_t2(int sign, std::array<IntType, DIM> modes, const std::complex<T>* in,
                    IntType num_out, std::array<const T*, DIM> out_points, std::complex<T>* out,
                    std::array<IntType, DIM> strides) {
  if constexpr (DIM == 1) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_2_d1<T>)
    (sign, modes, in , num_out, out_points, out, strides);
  } else if constexpr (DIM == 2) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_2_d2<T>)
    (sign, modes, in , num_out, out_points, out, strides);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_2_d3<T>)
    (sign, modes, in , num_out, out_points, out, strides);
  }
}

template <typename T, IntType DIM>
void nuft_direct_t1(int sign, std::array<IntType, DIM> modes, const std::complex<T>* in,
                    IntType num_in, std::array<const T*, DIM> in_points, std::complex<T>* out,
                    std::array<IntType, DIM> strides) {
  if constexpr (DIM == 1) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_1_d1<T>)
    (sign, modes, in, num_in, in_points, out, strides);
  } else if constexpr (DIM == 2) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_1_d2<T>)
    (sign, modes, in, num_in, in_points, out, strides);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_1_d3<T>)
    (sign, modes, in, num_in, in_points, out, strides);
  }
}

template <typename T, IntType DIM>
void nuft_direct_t3(int sign, IntType num_in, std::array<const T*, DIM> in_points,
                    const std::complex<T>* in, IntType num_out,
                    std::array<const T*, DIM> out_points, std::complex<T>* out) {
  if constexpr (std::is_same_v<float, T>) {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_3_float<DIM>)
    (sign, num_in, in_points, in, num_out, out_points, out);
  } else {
    NEONUFFT_EXPORT_AND_DISPATCH_T(type_3_double<DIM>)
    (sign, num_in, in_points, in, num_out, out_points, out);
  }
}

template void nuft_direct_t2<float, 1>(int sign, std::array<IntType, 1> modes,
                                       const std::complex<float>* in, IntType num_out,
                                       std::array<const float*, 1> out_points,
                                       std::complex<float>* out, std::array<IntType, 1> strides);

template void nuft_direct_t2<float, 2>(int sign, std::array<IntType, 2> modes,
                                       const std::complex<float>* in, IntType num_out,
                                       std::array<const float*, 2> out_points,
                                       std::complex<float>* out, std::array<IntType, 2> strides);

template void nuft_direct_t2<float, 3>(int sign, std::array<IntType, 3> modes,
                                       const std::complex<float>* in, IntType num_out,
                                       std::array<const float*, 3> out_points,
                                       std::complex<float>* out, std::array<IntType, 3> strides);

template void nuft_direct_t2<double, 1>(int sign, std::array<IntType, 1> modes,
                                       const std::complex<double>* in, IntType num_out,
                                       std::array<const double*, 1> out_points,
                                       std::complex<double>* out, std::array<IntType, 1> strides);

template void nuft_direct_t2<double, 2>(int sign, std::array<IntType, 2> modes,
                                       const std::complex<double>* in, IntType num_out,
                                       std::array<const double*, 2> out_points,
                                       std::complex<double>* out, std::array<IntType, 2> strides);

template void nuft_direct_t2<double, 3>(int sign, std::array<IntType, 3> modes,
                                       const std::complex<double>* in, IntType num_out,
                                       std::array<const double*, 3> out_points,
                                       std::complex<double>* out, std::array<IntType, 3> strides);

template void nuft_direct_t1<float, 1>(int sign, std::array<IntType, 1> modes,
                                       const std::complex<float>* in, IntType num_in,
                                       std::array<const float*, 1> in_points,
                                       std::complex<float>* out, std::array<IntType, 1> strides);

template void nuft_direct_t1<float, 2>(int sign, std::array<IntType, 2> modes,
                                       const std::complex<float>* in, IntType num_in,
                                       std::array<const float*, 2> in_points,
                                       std::complex<float>* out, std::array<IntType, 2> strides);

template void nuft_direct_t1<float, 3>(int sign, std::array<IntType, 3> modes,
                                       const std::complex<float>* in, IntType num_in,
                                       std::array<const float*, 3> in_points,
                                       std::complex<float>* out, std::array<IntType, 3> strides);

template void nuft_direct_t1<double, 1>(int sign, std::array<IntType, 1> modes,
                                       const std::complex<double>* in, IntType num_in,
                                       std::array<const double*, 1> in_points,
                                       std::complex<double>* out, std::array<IntType, 1> strides);

template void nuft_direct_t1<double, 2>(int sign, std::array<IntType, 2> modes,
                                       const std::complex<double>* in, IntType num_in,
                                       std::array<const double*, 2> in_points,
                                       std::complex<double>* out, std::array<IntType, 2> strides);

template void nuft_direct_t1<double, 3>(int sign, std::array<IntType, 3> modes,
                                       const std::complex<double>* in, IntType num_in,
                                       std::array<const double*, 3> in_points,
                                       std::complex<double>* out, std::array<IntType, 3> strides);

template void nuft_direct_t3<float, 1>(int sign, IntType num_in,
                                       std::array<const float*, 1> in_points,
                                       const std::complex<float>* in, IntType num_out,
                                       std::array<const float*, 1> out_points,
                                       std::complex<float>* out);

template void nuft_direct_t3<float, 2>(int sign, IntType num_in,
                                       std::array<const float*, 2> in_points,
                                       const std::complex<float>* in, IntType num_out,
                                       std::array<const float*, 2> out_points,
                                       std::complex<float>* out);

template void nuft_direct_t3<float, 3>(int sign, IntType num_in,
                                       std::array<const float*, 3> in_points,
                                       const std::complex<float>* in, IntType num_out,
                                       std::array<const float*, 3> out_points,
                                       std::complex<float>* out);

template void nuft_direct_t3<double, 1>(int sign, IntType num_in,
                                       std::array<const double*, 1> in_points,
                                       const std::complex<double>* in, IntType num_out,
                                       std::array<const double*, 1> out_points,
                                       std::complex<double>* out);

template void nuft_direct_t3<double, 2>(int sign, IntType num_in,
                                       std::array<const double*, 2> in_points,
                                       const std::complex<double>* in, IntType num_out,
                                       std::array<const double*, 2> out_points,
                                       std::complex<double>* out);

template void nuft_direct_t3<double, 3>(int sign, IntType num_in,
                                       std::array<const double*, 3> in_points,
                                       const std::complex<double>* in, IntType num_out,
                                       std::array<const double*, 3> out_points,
                                       std::complex<double>* out);
#endif

}  // namespace test
} // namespace neonufft
