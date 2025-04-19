#pragma once

#include <array>

#include "neonufft/config.h"

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"

namespace neonufft{
template <typename T>
void fseries_inverse(NeonufftKernelType kernel_type, const KernelParameters<T>& kernel_param,
                     IntType grid_size, T* fs);
}
