#pragma once

#include "neonufft/config.h"
// ---

#include <array>

#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/types.hpp"

namespace neonufft {
namespace gpu {
inline auto kernel_launch_grid(const api::DevicePropType& prop, const std::array<IntType, 3>& size,
                               const dim3& blockSize) -> dim3 {
  const std::array<IntType, 3> blockSizeT = {static_cast<IntType>(blockSize.x),
                                             static_cast<IntType>(blockSize.y),
                                             static_cast<IntType>(blockSize.z)};
  return dim3(static_cast<int>(std::min<IntType>((size[0] + blockSizeT[0] - 1) / blockSizeT[0],
                                                 prop.maxGridSize[0])),
              static_cast<int>(std::min<IntType>((size[1] + blockSizeT[1] - 1) / blockSizeT[1],
                                                 prop.maxGridSize[1])),
              static_cast<int>(std::min<IntType>((size[2] + blockSizeT[2] - 1) / blockSizeT[2],
                                                 prop.maxGridSize[2])));
}
}  // namespace gpu
}  // namespace neonufft
