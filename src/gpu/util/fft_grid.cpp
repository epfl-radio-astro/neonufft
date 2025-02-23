#include "gpu/util/fft_grid.hpp"

#include "neonufft/config.h"
//---

#include <array>
#include <cstddef>
#include <tuple>

#include "gpu/util/fft_api.hpp"

namespace neonufft {
namespace gpu {

template <typename T, std::size_t DIM>
FFTGrid<T, DIM>::FFTGrid() : plan_(nullptr, [](void *) {}) {}

template <typename T, std::size_t DIM>
FFTGrid<T, DIM>::FFTGrid(const std::shared_ptr<Allocator>& alloc, StreamType stream,
                         IndexType shape, int sign, IndexType padding)
    : plan_(nullptr, [](void*) {}), padding_(padding), sign_(sign) {
  auto paddedShape = shape;
  for (IntType d = 0; d < DIM; ++d) {
    paddedShape[d] += 2 * padding[d];
  }

  padded_grid_.reset(paddedShape, alloc);
  grid_ = padded_grid_.sub_view(padding, shape);

  // fftw is row-major. Stored grid is coloumn-major.
  std::array<int, DIM> n;
  if constexpr (DIM == 1) {
    n[0] = grid_.shape(0);
  } else if constexpr (DIM == 2){
    n[0] = grid_.shape(1);
    n[1] = grid_.shape(0);
  } else {
    n[0] = grid_.shape(2);
    n[1] = grid_.shape(1);
    n[2] = grid_.shape(0);
  }

  std::array<int, DIM> nembed;
  if constexpr (DIM == 1) {
    nembed[0] = grid_.shape(0);
  } else if constexpr (DIM == 2){
    nembed[0] = grid_.shape(1);
    nembed[1] = grid_.strides(1);
  } else {
    nembed[0] = grid_.shape(2);
    nembed[1] = grid_.strides(2) / grid_.strides(1);
    nembed[2] = grid_.strides(1);
  }

  int dist = grid_.size();
  int batch = 1;
  int stride = 1;

  fft::HandleType new_plan;
  std::size_t worksize = 0;

  // create plan
  fft::create(&new_plan);
  gpu::fft::set_auto_allocation(new_plan, 0);
  gpu::fft::make_plan_many(new_plan, DIM, n.data(), nembed.data(), stride, dist, nembed.data(),
                           stride, dist, gpu::fft::TransformType::ComplexToComplex<T>::value, batch,
                           &worksize);

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
