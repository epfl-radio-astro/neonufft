#pragma once

#include "neonufft/config.h"
//---

#include <array>

#include "neonufft/es_kernel_param.hpp"
#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/util/kernel_launch_grid.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/enums.h"
#include "neonufft/gpu/util/partition_group.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/util/point.hpp"

namespace neonufft {
namespace gpu {
template <typename T, IntType DIM>
void spread(const api::DevicePropType& prop, const api::StreamType& stream,
            const KernelParameters<T>& param, ConstDeviceView<PartitionGroup, DIM> partition,
            ConstDeviceView<Point<T, DIM>, 1> points, ConstDeviceView<ComplexType<T>, 1> input,
            ConstDeviceView<ComplexType<T>, 1> prephase_optional,
            DeviceView<ComplexType<T>, DIM> padded_grid);
}  // namespace gpu
}  // namespace neonufft
