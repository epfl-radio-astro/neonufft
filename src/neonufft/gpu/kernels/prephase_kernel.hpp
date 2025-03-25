#pragma once

#include "neonufft/config.h"
//---

#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/util/stack_array.hpp"

namespace neonufft {
namespace gpu {
template <typename T, IntType DIM>
auto compute_prephase(const api::DevicePropType& prop, const api::StreamType& stream, int sign,
                      StackArray<ConstDeviceView<T, 1>, DIM> input_points,
                      StackArray<T, DIM> out_offset, DeviceView<ComplexType<T>, 1> prephase)
    -> void;
}  // namespace gpu
}  // namespace neonufft
