#pragma once

#include <neonufft/config.h>
// ---

#include <neonufft/types.hpp>
#include <neonufft/enums.h>

#include <type_traits>
#include <cstddef>
#include <array>
#include <memory>
#include <complex>

/*! \cond PRIVATE */
namespace neonufft {
/*! \endcond */

struct NEONUFFT_EXPORT Options {
  double tol = 0.001;
  double upsampfac = 2.0;
  double recenter_threshold = 0.1;
  int num_threads = 0;
  bool sort_input = true;
  bool sort_output = true;
  bool kernel_approximation = false;
  NeonufftModeOrder order = NEONUFFT_MODE_ORDER_CMCL;
  NeonufftKernelType kernel_type = NEONUFFT_ES_KERNEL;
};

template <typename T, std::size_t DIM> class NEONUFFT_EXPORT Plan {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(DIM == 1 || DIM == 2 || DIM == 3);

  Plan(Options opt, int sign, IntType num_nu, std::array<const T *, DIM> loc,
       std::array<IntType, DIM> modes);

  void transform_type_1(const std::complex<T> *in, std::complex<T> *out,
                        std::array<IntType, DIM> out_strides);

  void transform_type_2(const std::complex<T> *in,
                        std::array<IntType, DIM> in_strides,
                        std::complex<T> *out);

  void set_modes(std::array<IntType, DIM> modes);

  void set_nu_points(IntType num_nu, std::array<const T *, DIM> loc);

  // TODO: update sign?

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
         std::array<const T *, DIM> output_points);

  PlanT3(Options opt, int sign, std::array<T, DIM> input_min,
         std::array<T, DIM> input_max, std::array<T, DIM> output_min,
         std::array<T, DIM> output_max);

  void set_input_points(IntType num_in,
                        std::array<const T *, DIM> input_points);

  void set_output_points(IntType num_out,
                         std::array<const T *, DIM> output_points);

  void add_input(const std::complex<T> *in);

  void transform(std::complex<T> *out);

private:
  std::unique_ptr<void, void (*)(void *)> impl_;
};

/*! \cond PRIVATE */
} // namespace neonufft
/*! \endcond */
