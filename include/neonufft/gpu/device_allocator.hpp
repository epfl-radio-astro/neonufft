#pragma once

#include <neonufft/config.h>
// ---

#include <cstddef>
#include <neonufft/allocator.hpp>

/*! \cond PRIVATE */
namespace neonufft {
namespace gpu {
/*! \endcond */

class DeviceAllocator : public Allocator {
public:
  DeviceAllocator() = default;

  void* allocate(std::size_t size) override;

  void deallocate(void* ptr) noexcept override;
};

/*! \cond PRIVATE */
}  // namespace gpu
}  // namespace neonufft
/*! \endcond */
