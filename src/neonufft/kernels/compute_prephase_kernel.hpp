#pragma once

#include <array>
#include <complex>
#include <utility>

#include "neonufft/config.h"

#include "neonufft/types.hpp"

namespace neonufft {

template <typename T, IntType DIM>
void compute_prephase(int sign, IntType n, std::array<const T *, DIM> in_loc,
                      std::array<T, DIM> out_offset,
                      std::complex<T> *prephase_aligned);

} // namespace neonufft
