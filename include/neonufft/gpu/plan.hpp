#pragma once

#include <neonufft/config.h>
// ---

#include <neonufft/enums.h>

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <neonufft/gpu/types.hpp>
#include <neonufft/plan.hpp>
#include <neonufft/types.hpp>
#include <type_traits>
#include <utility>

/*! \cond PRIVATE */
namespace neonufft {
namespace gpu {
/*! \endcond */

template <typename T, std::size_t DIM> class NEONUFFT_EXPORT Plan {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM == 1 || DIM == 2 || DIM == 3);


  Plan(Options opt, int sign, IntType num_nu, std::array<const T *, DIM> loc,
       std::array<IntType, DIM> modes, StreamType stream);

  void transform_type_1(const ComplexType<T> *in, ComplexType<T> *out,
                        std::array<IntType, DIM> out_strides);

  void transform_type_2(const ComplexType<T> *in,
                        std::array<IntType, DIM> in_strides,
                        ComplexType<T> *out);

  void set_modes(std::array<IntType, DIM> modes);

  void set_nu_points(IntType num_nu, std::array<const T *, DIM> loc);

private:
  std::unique_ptr<void, void (*)(void *)> impl_;
};

template <typename T, std::size_t DIM> class NEONUFFT_EXPORT PlanT3 {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM == 1 || DIM == 2 || DIM == 3);

  static std::uint64_t grid_memory_size(const Options &opt,
                                        std::array<T, DIM> input_min,
                                        std::array<T, DIM> input_max,
                                        std::array<T, DIM> output_min,
                                        std::array<T, DIM> output_max);

  PlanT3(Options opt, int sign, IntType num_in,
         std::array<const T *, DIM> input_points, IntType num_out,
         std::array<const T *, DIM> output_points, StreamType stream);

  PlanT3(Options opt, int sign, std::array<T, DIM> input_min,
         std::array<T, DIM> input_max, std::array<T, DIM> output_min,
         std::array<T, DIM> output_max, StreamType stream);

  void set_input_points(IntType num_in,
                        std::array<const T *, DIM> input_points);

  void set_output_points(IntType num_out,
                         std::array<const T *, DIM> output_points);

  void add_input(const ComplexType<T> *in);

  void transform(ComplexType<T> *out);

private:
  std::unique_ptr<void, void (*)(void *)> impl_;
};

/*! \cond PRIVATE */
} // namespace gpu
} // namespace neonufft
/*! \endcond */
