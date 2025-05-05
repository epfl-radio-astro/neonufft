#pragma once

#include "neonufft/config.h"
//---

#include "neonufft/enums.h"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"

namespace neonufft {
namespace gpu {
template <typename T>
auto fseries_inverse(const api::DevicePropType& prop, const api::StreamType& stream,
                     const KernelParameters<T>& param, IntType grid_size,
                     DeviceView<T, 1> fseries_inverse) -> void;
}  // namespace gpu
}  // namespace neonufft
