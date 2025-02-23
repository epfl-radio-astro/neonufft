#pragma once

#include "neonufft//config.h"
//---

#include <algorithm>
#include <array>

#include "es_kernel_param.hpp"
#include "gpu/memory/device_view.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime_api.hpp"
#include "neonufft/enums.h"
#include "neonufft/gpu/types.hpp"
#include "util/point.hpp"

namespace neonufft {
namespace gpu {
template <typename T, IntType DIM>
auto interpolation(const api::DevicePropType& prop, const api::StreamType& stream,
                   const KernelParameters<T>& param, ConstDeviceView<Point<T, DIM>, 1> points,
                   ConstDeviceView<ComplexType<T>, DIM> grid, DeviceView<ComplexType<T>, 1> out)
    -> void;
}  // namespace gpu
}  // namespace neonufft
