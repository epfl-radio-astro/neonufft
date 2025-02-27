#pragma once

#include "neonufft/config.h"
// ---

#include "neonufft/types.hpp"

namespace neonufft {
namespace gpu {
struct PartitionGroup {
  static constexpr inline int width = 8;

  unsigned long long begin = 0;
  int size = 0;
};
}  // namespace gpu
}  // namespace neonufft
