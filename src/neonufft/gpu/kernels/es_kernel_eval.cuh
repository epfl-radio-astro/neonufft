#pragma once

#include "neonufft//config.h"
//---

#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"

namespace neonufft {
namespace gpu {

template <typename T, int N_SPREAD>
struct EsKernelDirect;

template <int N_SPREAD>
struct EsKernelDirect<float, N_SPREAD> {
  constexpr static int n_spread = N_SPREAD;


  __device__ __forceinline__ float eval_scalar(float x) const {
    constexpr float es_c = 4.f / float(N_SPREAD * N_SPREAD);
    const float arg = 1.f - es_c * x * x;
    if (arg <= 0) return 0.0;

    // return expf(es_beta * (sqrtf(arg) - 1.f));
#ifdef NEONUFFT_ROCM
    return __expf(es_beta * (sqrtf(arg) - 1.f));
#else
    return __expf(es_beta * (__fsqrt_ru(arg) - 1.f));
#endif
  }

  float es_beta;
};

template <int N_SPREAD>
struct EsKernelDirect<double, N_SPREAD> {
  constexpr static int n_spread = N_SPREAD;

  __device__ __forceinline__ double eval_scalar(double x) const {
    constexpr float es_c = 4.0 / double(N_SPREAD * N_SPREAD);
    const double arg = 1.0 - es_c * x * x;
    if (arg <= 0) return 0.0;

    return exp(es_beta * (sqrt(arg) - 1.0));
  }

  double es_beta;
};

}  // namespace gpu
}  // namespace neonufft
