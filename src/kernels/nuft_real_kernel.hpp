#pragma once

#include <array>

#include "neonufft/config.h"

#include "es_kernel_param.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"

namespace neonufft{
template <typename T, IntType DIM>
void nuft_real(NeonufftKernelType kernel_type,
               const KernelParameters<T> &kernel_param, IntType num_in,
               std::array<const T *, DIM> loc, std::array<T, DIM> offsets,
               std::array<T, DIM> scaling_factors, T *phi_had_aligned);
}
