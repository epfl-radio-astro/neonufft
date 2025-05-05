#include "neonufft/config.h"
// ---
#include <tuple>

#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/gpu/device_allocator.hpp"

namespace neonufft {
namespace gpu {

void* DeviceAllocator::allocate(std::size_t size) {
  void* ptr = nullptr;
  if (size) api::malloc(&ptr, size);
  return ptr;
}

void DeviceAllocator::deallocate(void* ptr) noexcept {
  if (ptr) std::ignore = api::free(ptr);
}

} // namespace gpu
} // namespace neonufft
