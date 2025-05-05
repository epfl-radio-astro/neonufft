#pragma once

#include <complex>

#include "neonufft/config.h"

#include "neonufft/memory/view.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"

namespace neonufft{
template <typename T, IntType DIM>
void downsample(NeonufftModeOrder order, std::array<IntType, DIM> n_large,
                ConstHostView<std::complex<T>, DIM> in,
                std::array<IntType, DIM> n_small,
                std::array<const T *, DIM> ker,
                HostView<std::complex<T>, DIM> out);

} // namespace neonufft
