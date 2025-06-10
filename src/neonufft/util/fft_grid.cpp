#include "neonufft/config.h"
//---

#include <any>
#include <array>
#include <fftw3.h>
#include <mutex>
#include <tuple>

#include "neonufft/util/fft_grid.hpp"


namespace neonufft {

namespace {

struct FFTWInitGuard {
  FFTWInitGuard() {
    std::ignore = fftw_init_threads();
    std::ignore = fftwf_init_threads();
  }
};

FFTWInitGuard global_fft_init_guard;
std::mutex global_fftw_mutex;


template<typename T>
struct FFTW;

template <>
struct FFTW<double> {
  using ValueType = double;
  using ComplexType = fftw_complex;
  using PlanType = fftw_plan;

  template <typename... ARGS>
  static auto alignment_of(ARGS&&... args) -> int {
    return fftw_alignment_of(args...);
  }

  template <typename... ARGS>
  static auto plan_dft_1d(ARGS&&... args) -> fftw_plan {
    return fftw_plan_dft_1d(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft(ARGS&&... args) -> fftw_plan {
    return fftw_plan_many_dft(args...);
  }

  template <typename... ARGS>
  static auto destroy_plan(ARGS&&... args) -> void {
    fftw_destroy_plan(args...);
  }

  template <typename... ARGS>
  static auto execute(ARGS&&... args) -> void {
    fftw_execute(args...);
  }

  template <typename... ARGS>
  static auto execute_dft(ARGS&&... args) -> void {
    fftw_execute_dft(args...);
  }

  template <typename... ARGS>
  static auto plan_with_nthreads(ARGS&&... args) -> void {
    fftw_plan_with_nthreads(args...);
  }
};

template <>
struct FFTW<float> {
  using ValueType = float;
  using ComplexType = fftwf_complex;
  using PlanType = fftwf_plan;

  template <typename... ARGS>
  static auto alignment_of(ARGS&&... args) -> int {
    return fftwf_alignment_of(args...);
  }

  template <typename... ARGS>
  static auto plan_dft_1d(ARGS&&... args) -> fftwf_plan {
    return fftwf_plan_dft_1d(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft(ARGS&&... args) -> fftwf_plan {
    return fftwf_plan_many_dft(args...);
  }

  template <typename... ARGS>
  static auto destroy_plan(ARGS&&... args) -> void {
    fftwf_destroy_plan(args...);
  }

  template <typename... ARGS>
  static auto execute(ARGS&&... args) -> void {
    fftwf_execute(args...);
  }

  template <typename... ARGS>
  static auto execute_dft(ARGS&&... args) -> void {
    fftwf_execute_dft(args...);
  }

  template <typename... ARGS>
  static auto plan_with_nthreads(ARGS&&... args) -> void {
    fftwf_plan_with_nthreads(args...);
  }
};

} // namespace

template <typename T, std::size_t DIM>
FFTGrid<T, DIM>::FFTGrid() : plan_(nullptr, [](void *) {}) {}

template <typename T, std::size_t DIM>
FFTGrid<T, DIM>::FFTGrid(IntType num_threads, IndexType shape, int sign,
                         IndexType padding)
    : plan_(nullptr, [](void *) {}), padding_(padding) {

  auto paddedShape = shape;
  for (IntType d = 0; d < DIM; ++d) {
    paddedShape[d] += 2 * padding[d];
  }

  padded_grid_.reset(paddedShape);
  grid_ = padded_grid_.sub_view(padding, shape);

  // plan creation is not thread-safe
  std::lock_guard<std::mutex> guard(global_fftw_mutex);

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

  auto in_out = reinterpret_cast<typename FFTW<T>::ComplexType *>(grid_.data());

  // set threads
  FFTW<T>::plan_with_nthreads(num_threads);

  auto new_plan = FFTW<T>::plan_many_dft(
          DIM, n.data(), 1, in_out, nembed.data(), 1, 0, in_out, nembed.data(),
          1, 0, sign, FFTW_ESTIMATE);
  plan_ =
      decltype(plan_)(new typename FFTW<T>::PlanType(new_plan), [](void *ptr) {
        if (ptr) {
          std::lock_guard<std::mutex> guard(global_fftw_mutex);
          auto plan_ptr = reinterpret_cast<typename FFTW<T>::PlanType *>(ptr);
          FFTW<T>::destroy_plan(*plan_ptr);
          delete plan_ptr;
        }
      });

}

template <typename T, std::size_t DIM> void FFTGrid<T, DIM>::transform() {
  if (plan_) {
    auto plan_ptr = reinterpret_cast<typename FFTW<T>::PlanType *>(plan_.get());
    FFTW<T>::execute(*plan_ptr);
  }
}



template class FFTGrid<float, 1>;
template class FFTGrid<float, 2>;
template class FFTGrid<float, 3>;

template class FFTGrid<double, 1>;
template class FFTGrid<double, 2>;
template class FFTGrid<double, 3>;

} // namespace neonufft
