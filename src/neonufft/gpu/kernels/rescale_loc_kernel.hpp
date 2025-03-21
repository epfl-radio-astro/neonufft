#pragma once

#include "neonufft//config.h"
//---

#include <array>

#include "neonufft/enums.h"
#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/types.hpp"
#include "neonufft/gpu/util/partition_group.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/util/point.hpp"
#include "neonufft/util/stack_array.hpp"

namespace neonufft {
namespace gpu {
template <typename T, IntType DIM>
auto rescale_and_permut(const api::DevicePropType& prop, const api::StreamType& stream,
                        StackArray<ConstDeviceView<T, 1>, DIM> loc, IndexArray<DIM> grid_size,
                        DeviceView<PartitionGroup, DIM> partition,
                        DeviceView<Point<T, DIM>, 1> points) -> void;

template <typename T, IntType DIM>
auto rescale_and_permut_t3(const api::DevicePropType& prop, const api::StreamType& stream,
                           StackArray<ConstDeviceView<T, 1>, DIM> loc, IndexArray<DIM> grid_size,
                           StackArray<T, DIM> offset, StackArray<T, DIM> scaling_factor,
                           DeviceView<PartitionGroup, DIM> partition,
                           DeviceView<Point<T, DIM>, 1> points) -> void;

template <typename T, IntType DIM>
auto rescale_t3(const api::DevicePropType& prop, const api::StreamType& stream,
                StackArray<ConstDeviceView<T, 1>, DIM> loc, IndexArray<DIM> grid_size,
                StackArray<T, DIM> offset, StackArray<T, DIM> scaling_factor,
                DeviceView<Point<T, DIM>, 1> points) -> void;
}  // namespace gpu
}  // namespace neonufft
