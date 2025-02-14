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
void interpolate(NeonufftKernelType kernel_type,
                 const KernelParameters<T> &kernel_param,
                 ConstHostView<std::complex<T>, DIM> grid, IntType n_out,
                 const Point<T, DIM> *px, std::complex<T> *out);
}
