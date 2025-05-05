#pragma once

#include "neonufft/config.h"

#ifdef NEONUFFT_ROCM
#include <hipcub/hipcub.hpp>
#endif
#ifdef NEONUFFT_CUDA
#include <cub/cub.cuh>
#endif

namespace neonufft {
namespace gpu {
#ifdef NEONUFFT_CUDA
namespace cub_api = ::cub;
#endif
#ifdef NEONUFFT_ROCM
namespace cub_api = ::hipcub;
#endif
}  // namespace gpu
}  // namespace neonufft
