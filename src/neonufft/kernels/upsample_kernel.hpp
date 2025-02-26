#pragma once

#include <complex>

#include "neonufft/config.h"

#include "neonufft/memory/view.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"

namespace neonufft{
template <typename T, IntType DIM>
void upsample(NeonufftModeOrder order,
              ConstHostView<std::complex<T>, DIM> small_grid,
              std::array<ConstHostView<T, 1>, DIM> ker,
              HostView<std::complex<T>, DIM> large_grid);
} // namespace neonufft
