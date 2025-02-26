#pragma once

#include "neonufft/config.h"
//---

#include "neonufft/types.hpp"

namespace neonufft {

template<typename T, IntType DIM>
struct Point {
	T coord[DIM] = {0};
	IntType index = 0;
};
} // namespace neonufft
