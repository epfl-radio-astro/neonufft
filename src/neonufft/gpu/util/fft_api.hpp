#pragma once

#include "neonufft/config.h"

#if defined(NEONUFFT_CUDA)
#include <cufft.h>
#define GPU_FFT_PREFIX(val) cufft##val

#elif defined(NEONUFFT_ROCM)

#if __has_include(<hipfft/hipfft.h>)
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif

#define GPU_FFT_PREFIX(val) hipfft##val
#endif

// only declare namespace members if GPU support is enabled
#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)

#include <utility>

#include "neonufft/exceptions.hpp"
#include "neonufft/gpu/types.hpp"

namespace neonufft {
namespace gpu {
namespace fft {

// ==================================
// Types
// ==================================
using ResultType = GPU_FFT_PREFIX(Result);
using HandleType = GPU_FFT_PREFIX(Handle);
using FFTComplexFloatType = GPU_FFT_PREFIX(Complex);
using FFTComplexDoubleType = GPU_FFT_PREFIX(DoubleComplex);

// ==================================
// Transform types
// ==================================
namespace TransformDirection {
#ifdef NEONUFFT_CUDA
constexpr auto Forward = CUFFT_FORWARD;
constexpr auto Backward = CUFFT_INVERSE;
#else
constexpr auto Forward = HIPFFT_FORWARD;
constexpr auto Backward = HIPFFT_BACKWARD;
#endif
}  // namespace TransformDirection

// ==================================
// Transform types
// ==================================
namespace TransformType {
#ifdef NEONUFFT_CUDA
constexpr auto R2C = CUFFT_R2C;
constexpr auto C2R = CUFFT_C2R;
constexpr auto C2C = CUFFT_C2C;
constexpr auto D2Z = CUFFT_D2Z;
constexpr auto Z2D = CUFFT_Z2D;
constexpr auto Z2Z = CUFFT_Z2Z;
#else
constexpr auto R2C = HIPFFT_R2C;
constexpr auto C2R = HIPFFT_C2R;
constexpr auto C2C = HIPFFT_C2C;
constexpr auto D2Z = HIPFFT_D2Z;
constexpr auto Z2D = HIPFFT_Z2D;
constexpr auto Z2Z = HIPFFT_Z2Z;
#endif

// Transform type selector
template <typename T>
struct ComplexToComplex;

template <>
struct ComplexToComplex<double> {
  constexpr static auto value = Z2Z;
};

template <>
struct ComplexToComplex<float> {
  constexpr static auto value = C2C;
};

// Transform type selector
template <typename T>
struct RealToComplex;

template <>
struct RealToComplex<double> {
  constexpr static auto value = D2Z;
};

template <>
struct RealToComplex<float> {
  constexpr static auto value = R2C;
};

// Transform type selector
template <typename T>
struct ComplexToReal;

template <>
struct ComplexToReal<double> {
  constexpr static auto value = Z2D;
};

template <>
struct ComplexToReal<float> {
  constexpr static auto value = C2R;
};
}  // namespace TransformType

// ==================================
// Result values
// ==================================
namespace result {
#ifdef NEONUFFT_CUDA
constexpr auto Success = CUFFT_SUCCESS;
#else
constexpr auto Success = HIPFFT_SUCCESS;
#endif
}  // namespace result

// ==================================
// Error check functions
// ==================================

inline auto check_result(ResultType error) -> void {
  if (error != result::Success) {
    throw GPUFFTError();
  }
}

// ==================================
// Execution function overload
// ==================================
inline auto execute(HandleType& plan, ComplexType<double>* data, int direction) -> void {
  check_result(GPU_FFT_PREFIX(ExecZ2Z)(plan, reinterpret_cast<FFTComplexDoubleType*>(data),
                                       reinterpret_cast<FFTComplexDoubleType*>(data), direction));
}

inline auto execute(HandleType& plan, ComplexType<float>* data, int direction) -> void {
  check_result(GPU_FFT_PREFIX(ExecC2C)(plan, reinterpret_cast<FFTComplexFloatType*>(data),
                                       reinterpret_cast<FFTComplexFloatType*>(data), direction));
}

// ==================================
// Forwarding functions of to GPU API
// ==================================
template <typename... ARGS>
inline auto create(ARGS&&... args) -> void {
  check_result(GPU_FFT_PREFIX(Create)(std::forward<ARGS>(args)...));
}

template <typename... ARGS>
inline auto make_plan_many(ARGS&&... args) -> void {
  check_result(GPU_FFT_PREFIX(MakePlanMany)(std::forward<ARGS>(args)...));
}

template <typename... ARGS>
inline auto set_work_area(ARGS&&... args) -> void {
  check_result(GPU_FFT_PREFIX(SetWorkArea)(std::forward<ARGS>(args)...));
}

template <typename... ARGS>
inline auto destroy(ARGS&&... args) -> ResultType {
  return GPU_FFT_PREFIX(Destroy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_stream(ARGS&&... args) -> void {
  check_result(GPU_FFT_PREFIX(SetStream)(std::forward<ARGS>(args)...));
}

template <typename... ARGS>
inline auto set_auto_allocation(ARGS&&... args) -> void {
  check_result(GPU_FFT_PREFIX(SetAutoAllocation)(std::forward<ARGS>(args)...));
}

}  // namespace fft
}  // namespace gpu
}  // namespace neonufft

#undef GPU_FFT_PREFIX

#endif  // defined NEONUFFT_CUDA || NEONUFFT_ROCM
