#pragma once

#include <complex>

#include "neonufft/config.h"

#include "neonufft/memory/view.hpp"
#include "neonufft/enums.h"
#include "neonufft/types.hpp"

namespace neonufft {
namespace test {
template <typename T, IntType DIM>
void nuft_direct_t2(int sign, std::array<IntType, DIM> modes, const std::complex<T>* in,
                    IntType num_out, std::array<const T*, DIM> out_points, std::complex<T>* out,
                    std::array<IntType, DIM> strides);

template <typename T, IntType DIM>
void nuft_direct_t1(int sign, std::array<IntType, DIM> modes, const std::complex<T>* in,
                    IntType num_in, std::array<const T*, DIM> in_points, std::complex<T>* out,
                    std::array<IntType, DIM> strides);

template <typename T, IntType DIM>
void nuft_direct_t3(int sign, IntType num_in, std::array<const T*, DIM> in_points,
                    const std::complex<T>* in, IntType num_out,
                    std::array<const T*, DIM> out_points, std::complex<T>* out);
}
}  // namespace neonufft
