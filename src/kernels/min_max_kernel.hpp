#pragma once

#include <utility>

#include "neonufft/config.h"

#include "neonufft/types.hpp"

namespace neonufft{
std::pair<float, float> min_max(IntType n, const float *input);

std::pair<double, double> min_max(IntType n, const double *input);
} // namespace neonufft
