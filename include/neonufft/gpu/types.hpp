#pragma once

#include <neonufft/config.h>
// ---

#include <type_traits>

#if defined(NEONUFFT_CUDA)
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#elif defined(NEONUFFT_ROCM)
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#endif

/*! \cond PRIVATE */
namespace neonufft {
namespace gpu {
/*! \endcond */

#if defined(NEONUFFT_CUDA)
using ComplexDoubleType = cuDoubleComplex;
using ComplexFloatType = cuComplex;
using StreamType = cudaStream_t;
#elif defined(NEONUFFT_ROCM)
using ComplexDoubleType = hipDoubleComplex;
using ComplexFloatType = hipComplex;
using StreamType = hipStream_t;
#endif

template <typename T>
using ComplexType =
    std::conditional_t<std::is_same_v<T, float>, ComplexFloatType, ComplexDoubleType>;

/*! \cond PRIVATE */
}  // namespace gpu
}  // namespace neonufft
/*! \endcond */
