#include "neonufft//config.h"
//---

#include <array>
#include <cstddef>

#include "neonufft/gpu/memory/device_view.hpp"
#include "neonufft/gpu/util/cub_api.hpp"
#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"

namespace neonufft {
namespace gpu {


template <typename T>
auto min_max_worksize(IntType size) -> IntType {
  std::size_t worksize_min = 0;
  std::size_t worksize_max = 0;
  api::check_status(
      cub_api::DeviceReduce::Min<const T*, T*>(nullptr, worksize_min, nullptr, nullptr, size, nullptr));
  api::check_status(
      cub_api::DeviceReduce::Max<const T*, T*>(nullptr, worksize_max, nullptr, nullptr, size, nullptr));

  return worksize_min > worksize_max ? worksize_min : worksize_max;
}

template auto min_max_worksize<float>(IntType size) -> IntType;
template auto min_max_worksize<double>(IntType size) -> IntType;

template <typename T>
auto min_max(ConstDeviceView<T, 1> input, T* device_min, T* device_max, DeviceView<T, 1> workbuffer,
             const api::StreamType& stream) -> void {
  std::size_t worksize = workbuffer.shape(0);
  api::check_status(cub_api::DeviceReduce::Min<const T*, T*>(
      workbuffer.data(), worksize, input.data(), device_min, input.shape(0), stream));
  api::check_status(cub_api::DeviceReduce::Max<const T*, T*>(
      workbuffer.data(), worksize, input.data(), device_max, input.shape(0), stream));
}

template auto min_max<float>(ConstDeviceView<float, 1> input, float* device_min, float* device_max,
                             DeviceView<float, 1> workbuffer, const api::StreamType& stream)
    -> void;
template auto min_max<double>(ConstDeviceView<double, 1> input, double* device_min,
                              double* device_max, DeviceView<double, 1> workbuffer,
                              const api::StreamType& stream) -> void;

}  // namespace gpu
}  // namespace neonufft
