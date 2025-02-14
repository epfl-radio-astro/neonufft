#pragma once

#include <complex>

#include "neonufft/config.h"

#include "memory/view.hpp"
#include "neonufft/types.hpp"

namespace neonufft{
template <typename T, IntType DIM>
void fold_padding(IntType n_spread, HostView<std::complex<T>, DIM> padded_grid);
} // namespace neonufft
