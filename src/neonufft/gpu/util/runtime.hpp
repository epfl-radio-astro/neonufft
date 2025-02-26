#pragma once

#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/config.h"

#ifdef NEONUFFT_ROCM
#include <hip/hip_runtime.h>
#endif
#ifdef NEONUFFT_CUDA
#include <cuda_runtime.h>
#endif

namespace neonufft {
namespace gpu {
namespace api {

template <typename F, typename... ARGS>
inline auto launch_kernel(F func, const dim3 thread_grid, const dim3 thread_block,
                          const size_t shared_memory_bytes, const StreamType stream, ARGS&&... args)
    -> void {
#ifdef NEONUFFT_CUDA
  func<<<thread_grid, thread_block, shared_memory_bytes, stream>>>(std::forward<ARGS>(args)...);
#elif defined(NEONUFFT_ROCM)
  hipLaunchKernelGGL(func, thread_grid, thread_block, shared_memory_bytes, stream,
                     std::forward<ARGS>(args)...);
#endif
}

}  // namespace api
}  // namespace gpu
}  // namespace bipp
