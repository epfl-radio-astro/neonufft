#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <random>

#include "neonufft/config.h"

#include "kernels/compute_prephase_kernel.hpp"
#include "memory/array.hpp"
#include "neonufft/types.hpp"
#include "gtest/gtest.h"

using namespace neonufft;

namespace ref {
template <typename T, std::size_t DIM>
void compute_prephase(int sign, IntType n, std::array<const T *, DIM> in_loc,
                      std::array<T, DIM> out_offset,
                      std::complex<T> *prephase) {
  std::complex<T> imasign(0, sign);
  for (IntType j = 0; j < n; ++j) { // ... loop over src NU locs
    T phase = out_offset[0] * in_loc[0][j];
    if constexpr (DIM > 1)
      phase += out_offset[1] * in_loc[1][j];
    if constexpr (DIM > 2)
      phase += out_offset[2] * in_loc[2][j];
    prephase[j] = std::cos(phase) + imasign * std::sin(phase); // Euler
                                                               // e^{+-i.phase}
  }
}
}


template <typename T>
class PrephaseTest : public testing::Test {
 public:
   void test_ref(IntType dim, int sign, IntType n) {

     std::vector<T> input_1(n);
     std::vector<T> input_2;
     std::vector<T> input_3;

     std::minstd_rand rand_gen(42);
     std::uniform_real_distribution<T> rand_dist(-3, 3);

     for (auto &val : input_1) {
       val = rand_dist(rand_gen);
     }
     if (dim > 1) {
       input_2.resize(n);
       for (auto &val : input_2) {
         val = rand_dist(rand_gen);
       }
     }
     if (dim > 2) {
       input_3.resize(n);
       for (auto &val : input_3) {
         val = rand_dist(rand_gen);
       }
     }

     // must be algined for vectorized kernel
     HostArray<std::complex<T>, 1> prephase(n);
     std::vector<std::complex<T>> prephase_ref(n);

     if (dim == 1) {
       std::array<const T *, 1> in_loc = {input_1.data()};
       std::array<T, 1> out_offset = {1.0};
       ref::compute_prephase<T, 1>(sign, n, in_loc, out_offset,
                                   prephase_ref.data());
       compute_prephase<T, 1>(sign, n, in_loc, out_offset, prephase.data());
     }

     if (dim == 2) {
       std::array<const T *, 2> in_loc = {input_1.data(), input_2.data()};
       std::array<T, 2> out_offset = {1.0, 2.0};
       ref::compute_prephase<T, 2>(sign, n, in_loc, out_offset,
                                   prephase_ref.data());
       compute_prephase<T, 2>(sign, n, in_loc, out_offset, prephase.data());
     }

     if (dim == 3) {
       std::array<const T *, 3> in_loc = {input_1.data(), input_2.data(),
                                          input_3.data()};
       std::array<T, 3> out_offset = {1.0, 2.0, 3.0};
       ref::compute_prephase<T, 3>(sign, n, in_loc, out_offset,
                                   prephase_ref.data());
       compute_prephase<T, 3>(sign, n, in_loc, out_offset, prephase.data());
     }

     for(IntType i = 0; i < n; ++i) {
       EXPECT_NEAR(prephase[i].real(), prephase_ref[i].real(),
                   (T)std::abs(prephase_ref[i].real() * 0.01));
       EXPECT_NEAR(prephase[i].imag(), prephase_ref[i].imag(),
                   (T)std::abs(prephase_ref[i].imag() * 0.01));
     }

   }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PrephaseTest, TestTypes);

TYPED_TEST(PrephaseTest, OneDim13) { this->test_ref(1, 1, 13); }

TYPED_TEST(PrephaseTest, TwoDim24) { this->test_ref(2, -1, 24); }

TYPED_TEST(PrephaseTest, ThreeDim100) { this->test_ref(3, -1, 100); }
