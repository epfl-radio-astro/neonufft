#include "neonufft/gpu/util/fft_grid.hpp"

#include "neonufft/config.h"
//---

#include <array>
#include <cstddef>
#include <tuple>

#include "neonufft/gpu/util/fft_api.hpp"

namespace neonufft {
namespace gpu {

template <typename T, std::size_t DIM>
FFTGrid<T, DIM>::FFTGrid() : plan_(nullptr, [](void *) {}) {}

template <typename T, std::size_t DIM>
FFTGrid<T, DIM>::FFTGrid(const std::shared_ptr<Allocator>& alloc, StreamType stream,
                         IndexType shape, int sign)
    : plan_(nullptr, [](void*) {}), sign_(sign) {
  grid_.reset(shape, alloc);


  fft::HandleType new_plan;
  std::size_t worksize = 0;

  // create plan
  // fftw is row-major. Stored grid is coloumn-major.
  fft::create(&new_plan);
  gpu::fft::set_auto_allocation(new_plan, 0);
  if constexpr (DIM == 1) {
    gpu::fft::make_plan_1d(new_plan, grid_.shape(0),
                           gpu::fft::TransformType::ComplexToComplex<T>::value, 1, &worksize);
  } else if constexpr (DIM == 2) {
    gpu::fft::make_plan_2d(new_plan, grid_.shape(1), grid_.shape(0),
                           gpu::fft::TransformType::ComplexToComplex<T>::value, &worksize);
  } else {
    gpu::fft::make_plan_3d(new_plan, grid_.shape(2), grid_.shape(1), grid_.shape(0),
                           gpu::fft::TransformType::ComplexToComplex<T>::value, &worksize);
  }

  // set stream
  gpu::fft::set_stream(new_plan, stream);

  // set work area
  workspace_.reset(worksize, alloc);
  fft::set_work_area(new_plan, workspace_.data());

  plan_ = decltype(plan_)(new typename fft::HandleType(new_plan), [](void* ptr) {
    if (ptr) {
      auto plan_ptr = reinterpret_cast<fft::HandleType*>(ptr);
      std::ignore = fft::destroy(*plan_ptr);
      delete plan_ptr;
    }
  });
}

template <typename T, std::size_t DIM> void FFTGrid<T, DIM>::transform() {
  if (plan_) {
    auto plan_ptr = reinterpret_cast<fft::HandleType*>(plan_.get());

    auto dir =
        sign_ < 0 ? gpu::fft::TransformDirection::Forward : gpu::fft::TransformDirection::Backward;

    fft::execute(*plan_ptr, grid_.data(), dir);
  } else {
    throw InternalError("GPU FFT Grid not initialized.");
  }
}



template class FFTGrid<float, 1>;
template class FFTGrid<float, 2>;
template class FFTGrid<float, 3>;

template class FFTGrid<double, 1>;
template class FFTGrid<double, 2>;
template class FFTGrid<double, 3>;

}  // namespace gpu
} // namespace neonufft
