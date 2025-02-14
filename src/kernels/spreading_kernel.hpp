#pragma once

#include <complex>

#include "neonufft/config.h"

#include "es_kernel_param.hpp"
#include "memory/view.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"
#include "util/point.hpp"

namespace neonufft{
template <typename T, IntType DIM>
void spread(NeonufftKernelType kernel_type,
            const KernelParameters<T> &kernel_param, IntType num_nu,
            const Point<T, DIM> *px, const std::complex<T> *input,
            const std::complex<T> *prephase_optional,
            std::array<IntType, DIM> grid_size,
            HostView<std::complex<T>, DIM> grid);
} // namespace neonufft
