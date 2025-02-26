#pragma once

#include "neonufft/config.h"
//---

#include "neonufft/types.hpp"

namespace neonufft {

inline constexpr IntType spread_padding(IntType n_spread) {
  return n_spread / 2 + 2;
}

} // namespace neonufft
