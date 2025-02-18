#pragma once

#include <neonufft/config.h>
// ---

#include <cstddef>

/*! \cond PRIVATE */
namespace neonufft {
/*! \endcond */

class Allocator {
public:
  Allocator() = default;

  virtual ~Allocator() = default;

  virtual void* allocate(std::size_t size)  = 0;

  virtual void deallocate(void* ptr) noexcept = 0;
};

/*! \cond PRIVATE */
} // namespace neonufft
/*! \endcond */
