#pragma once

#include "neonufft/config.h"
//---

#include "neonufft/enums.h"
#include "neonufft/es_kernel_param.hpp"
#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/util/stack_array.hpp"

namespace neonufft {
namespace gpu {
template <typename T, IntType DIM>
auto postphase(const api::DevicePropType& prop, const api::StreamType& stream,
               const KernelParameters<T>& param, T sign, StackArray<ConstDeviceView<T, 1>, DIM> loc,
               StackArray<T, DIM> in_offsets, StackArray<T, DIM> out_offsets,
               StackArray<T, DIM> scaling_factors, DeviceView<ComplexType<T>, 1> postphase) -> void;
}  // namespace gpu
}  // namespace neonufft
