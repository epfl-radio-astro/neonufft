#pragma once

#include "neonufft//config.h"
//---

#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"

namespace neonufft {
namespace gpu {

template <typename T>
auto min_max_worksize(IntType size) -> IntType;

template <typename T>
auto min_max(ConstDeviceView<T, 1> input, T* device_min, T* device_max, DeviceView<T, 1> workbuffer,
             const api::StreamType& stream) -> void;

}  // namespace gpu
}  // namespace neonufft
