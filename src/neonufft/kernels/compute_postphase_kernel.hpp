#pragma once

#include <array>
#include <complex>
#include <utility>

#include "neonufft/config.h"

#include "neonufft/types.hpp"

namespace neonufft {
template <typename T, IntType DIM>
void compute_postphase(int sign, IntType n, const T *phi_hat_aligned,
                       std::array<const T *, DIM> out_loc,
                       std::array<T, DIM> in_offset,
                       std::array<T, DIM> out_offset,
                       std::complex<T> *postphase_aligned);

} // namespace neonufft
