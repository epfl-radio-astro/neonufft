#include "neonufft/config.h"

#include "neonufft/plan.hpp"
#include "neonufft/plan_impl.hpp"
#include "neonufft/plan_t3_impl.hpp"
#include <memory>

namespace neonufft {

template <typename T, std::size_t DIM>
Plan<T, DIM>::Plan(Options opt, int sign, IntType num_nu,
                   std::array<const T *, DIM> loc,
                   std::array<IntType, DIM> modes)
    : impl_(decltype(impl_)(
          new PlanImpl<T, DIM>(opt, sign, num_nu, loc, modes), [](void *ptr) {
            if (ptr) {
              delete reinterpret_cast<PlanImpl<T, DIM> *>(ptr);
            }
          })) {}

template <typename T, std::size_t DIM>
void Plan<T, DIM>::transform_type_1(const std::complex<T> *in,
                                    std::complex<T> *out,
                                    std::array<IntType, DIM> out_strides) {
  reinterpret_cast<PlanImpl<T, DIM> *>(impl_.get())
      ->transform_type_1(in, out, out_strides);
}

template <typename T, std::size_t DIM>
void Plan<T, DIM>::transform_type_2(const std::complex<T> *in,
                                    std::array<IntType, DIM> in_strides,
                                    std::complex<T> *out) {
  reinterpret_cast<PlanImpl<T, DIM> *>(impl_.get())
      ->transform_type_2(in, in_strides, out);
}

template <typename T, std::size_t DIM>
void Plan<T, DIM>::set_modes(std::array<IntType, DIM> modes) {
  reinterpret_cast<PlanImpl<T, DIM> *>(impl_.get())->set_modes(modes);
}

template <typename T, std::size_t DIM>
void Plan<T, DIM>::set_nu_points(IntType num_nu,
                                       std::array<const T *, DIM> loc) {
  reinterpret_cast<PlanImpl<T, DIM> *>(impl_.get())
      ->set_nu_points(num_nu, loc);
}

template class Plan<float, 1>;
template class Plan<float, 2>;
template class Plan<float, 3>;

template class Plan<double, 1>;
template class Plan<double, 2>;
template class Plan<double, 3>;

template <typename T, std::size_t DIM>
std::uint64_t PlanT3<T, DIM>::grid_memory_size(const Options &opt,
                                               std::array<T, DIM> input_min,
                                               std::array<T, DIM> input_max,
                                               std::array<T, DIM> output_min,
                                               std::array<T, DIM> output_max) {
  return PlanT3Impl<T, DIM>::grid_memory_size(opt, input_min, input_max,
                                              output_min, output_max);
}

template <typename T, std::size_t DIM>
PlanT3<T, DIM>::PlanT3(Options opt, int sign, IntType num_in,
                       std::array<const T *, DIM> input_points, IntType num_out,
                       std::array<const T *, DIM> output_points)
    : impl_(decltype(impl_)(new PlanT3Impl<T, DIM>(opt, sign, num_in,
                                                   input_points, num_out,
                                                   output_points),
                            [](void *ptr) {
                              if (ptr) {
                                delete reinterpret_cast<PlanT3Impl<T, DIM> *>(
                                    ptr);
                              }
                            })) {}

template <typename T, std::size_t DIM>
PlanT3<T, DIM>::PlanT3(Options opt, int sign, std::array<T, DIM> input_min,
                       std::array<T, DIM> input_max,
                       std::array<T, DIM> output_min,
                       std::array<T, DIM> output_max)
    : impl_(decltype(impl_)(
          new PlanT3Impl<T, DIM>(opt, sign, input_min, input_max, output_min,
                                 output_max),
          [](void *ptr) {
            if (ptr) {
              delete reinterpret_cast<PlanT3Impl<T, DIM> *>(ptr);
            }
          })) {}

template <typename T, std::size_t DIM>
void PlanT3<T, DIM>::transform( std::complex<T> *out) {
  reinterpret_cast<PlanT3Impl<T, DIM> *>(impl_.get())->transform(out);
}

template <typename T, std::size_t DIM>
void PlanT3<T, DIM>::add_input(const std::complex<T> *in) {
  reinterpret_cast<PlanT3Impl<T, DIM> *>(impl_.get())->add_input(in);
}

template <typename T, std::size_t DIM>
  void PlanT3<T, DIM>::set_input_points(IntType num_in,
                              std::array<const T *, DIM> input_points){
  reinterpret_cast<PlanT3Impl<T, DIM> *>(impl_.get())
      ->set_input_points(num_in, input_points);
}

template <typename T, std::size_t DIM>
void PlanT3<T, DIM>::set_output_points(
    IntType num_out, std::array<const T *, DIM> output_points) {
  reinterpret_cast<PlanT3Impl<T, DIM> *>(impl_.get())
      ->set_output_points(num_out, output_points);
}

template class PlanT3<float, 1>;
template class PlanT3<float, 2>;
template class PlanT3<float, 3>;

template class PlanT3<double, 1>;
template class PlanT3<double, 2>;
template class PlanT3<double, 3>;


} // namespace neonufft
