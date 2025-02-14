#include <cstddef>
#include <cstdlib>

#include "neonufft/config.h"

#include "memory/array.hpp"
#include "neonufft/exceptions.hpp"


namespace neonufft {
namespace memory {

auto allocate_aligned(std::size_t size) -> void * {
  const auto overhang = size % NEONUFFT_ALIGNMENT;
  const auto alloc_size =
      size + (overhang > 0) * (NEONUFFT_ALIGNMENT - overhang);
  auto ptr = std::aligned_alloc(NEONUFFT_ALIGNMENT, alloc_size);
  if (!ptr) {
    throw MemoryAllocError();
  }
  return ptr;
}

auto deallocate_aligned(void *ptr) -> void { std::free(ptr); }

auto padding_for_vectorization(std::size_t type_size,
                               std::size_t size) -> std::size_t {
  const auto size_in_bytes = type_size * size;
  const auto overhang_in_bytes = size_in_bytes % NEONUFFT_MAX_VEC_LENGTH;
  const auto padding_in_bytes =
      overhang_in_bytes > 0 ? 2 * NEONUFFT_MAX_VEC_LENGTH - overhang_in_bytes
                            : NEONUFFT_MAX_VEC_LENGTH;

  return padding_in_bytes / type_size;
}
} // namespace memory
} // namespace neonufft
