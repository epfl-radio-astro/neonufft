#pragma once

#include "neonufft/config.h"

#include "neonufft/types.hpp"

namespace neonufft {
struct PartitionGroup {
  IntType begin = 0;
  IntType size = 0;
};
} // namespace neonufft
