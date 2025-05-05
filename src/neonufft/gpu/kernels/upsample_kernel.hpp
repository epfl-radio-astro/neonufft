#pragma once

#include "neonufft/config.h"
//---

#include <algorithm>
#include <array>

#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/util/kernel_launch_grid.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/enums.h"
#include "neonufft/gpu/types.hpp"

namespace neonufft {
namespace gpu {

template <typename T, IntType DIM>
auto upsample(const api::DevicePropType& prop, const api::StreamType& stream,
              NeonufftModeOrder order, ConstDeviceView<ComplexType<T>, DIM> small_grid,
              std::array<ConstDeviceView<T, 1>, DIM> ker,
              DeviceView<ComplexType<T>, DIM> large_grid) -> void;

}  // namespace gpu
}  // namespace neonufft
