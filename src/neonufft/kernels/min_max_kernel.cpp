#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "neonufft/config.h"

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"
#include "neonufft/kernels/min_max_kernel.hpp"
#include "neonufft/util/math.hpp"
// #include "neonufft/kernels/upsample_kernel.hpp"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "neonufft/kernels/min_max_kernel.cpp" // this file

#include "neonufft/kernels/hwy_dispatch.hpp"

namespace neonufft {
namespace {

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR std::pair<T, T> min_max_kernel(IntType n,
                                        const T *HWY_RESTRICT input) {

  const TagType<T> d;
  const IntType n_lanes = hn::Lanes(d);

  auto v_min_1 = hn::Set(d, std::numeric_limits<T>::max());
  auto v_min_2 = hn::Set(d, std::numeric_limits<T>::max());
  auto v_max_1 = hn::Set(d, std::numeric_limits<T>::lowest());
  auto v_max_2 = hn::Set(d, std::numeric_limits<T>::lowest());

  IntType idx = 0;
  for (; idx + 2 * n_lanes <= n; idx += 2 * n_lanes) {
    const auto v_in_1 = hn::LoadU(d, input + idx);
    const auto v_in_2 = hn::LoadU(d, input + idx + n_lanes);

    v_min_1 = hn::Min(v_min_1, v_in_1);
    v_min_2 = hn::Min(v_min_2, v_in_2);

    v_max_1 = hn::Max(v_max_1, v_in_1);
    v_max_2 = hn::Max(v_max_2, v_in_2);
  }

  auto s_min =
      std::min<T>(hn::ReduceMin(d, v_min_1), hn::ReduceMin(d, v_min_2));
  auto s_max =
      std::max<T>(hn::ReduceMax(d, v_max_1), hn::ReduceMax(d, v_max_2));

  for (; idx < n; ++idx) {
    s_min = std::min<T>(input[idx], s_min);
    s_max = std::max<T>(input[idx], s_max);
  }

  return std::make_pair(s_min, s_max);
}

HWY_ATTR std::pair<float, float> min_max_float(IntType n, const float *input) {
  return min_max_kernel<float>(n, input);
}

HWY_ATTR std::pair<double, double> min_max_double(IntType n,
                                                  const double *input) {
  return min_max_kernel<double>(n, input);
}

} // namespace HWY_NAMESPACE
} // namespace

#if HWY_ONCE
NEONUFFT_EXPORT_FUNC(min_max_float);
NEONUFFT_EXPORT_FUNC(min_max_double);

std::pair<float, float> min_max(IntType n, const float *input) {
  return NEONUFFT_DISPATCH(min_max_float)(n, input);
}

std::pair<double, double> min_max(IntType n, const double *input) {
  return NEONUFFT_DISPATCH(min_max_double)(n, input);
}

#endif

} // namespace neonufft
