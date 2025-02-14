#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <random>

#include "neonufft/config.h"

#include "kernels/compute_postphase_kernel.hpp"
#include "memory/array.hpp"
#include "neonufft/types.hpp"
#include "gtest/gtest.h"

using namespace neonufft;

namespace ref {
template <typename T, IntType DIM>
void compute_postphase(int sign, IntType n, const T *phi_hat,
                       std::array<const T *, DIM> out_loc,
                       std::array<T, DIM> in_offset,
                       std::array<T, DIM> out_offset,
                       std::complex<T> *postphase) {
  for (IntType k = 0; k < n; ++k) { // .... loop over NU targ freqs
    postphase[k] = std::complex<T>(1.0 / phi_hat[k], 0);
    T phase = (out_loc[0][k] - out_offset[0]) * in_offset[0];
    if constexpr (DIM > 1)
      phase += (out_loc[1][k] - out_offset[1]) * in_offset[1];
    if constexpr (DIM > 2)
      phase += (out_loc[2][k] - out_offset[2]) * in_offset[2];

    postphase[k] *= std::complex<T>(
        std::cos(phase), sign * std::sin(phase)); // Euler e^{+-i.phase}
  }
}
} // namespace ref

template <typename T>
class PostphaseTest : public testing::Test {
 public:
   void test_ref(IntType dim, int sign, IntType n) {


     std::vector<T> output_loc1(n);
     std::vector<T> output_loc2;
     std::vector<T> output_loc3;

     std::minstd_rand rand_gen(42);
     std::uniform_real_distribution<T> rand_dist(-3, 3);

     for (auto &val : output_loc1) {
       val = rand_dist(rand_gen);
     }
     if (dim > 1) {
       output_loc2.resize(n);
       for (auto &val : output_loc2) {
         val = rand_dist(rand_gen);
       }
     }
     if (dim > 2) {
       output_loc3.resize(n);
       for (auto &val : output_loc3) {
         val = rand_dist(rand_gen);
       }
     }

     // must be aligned
     HostArray<T, 1> phi_hat(n);
     for (IntType i = 0; i < n; ++i) {
       phi_hat[i] = rand_dist(rand_gen);
     }

     // must be algined for vectorized kernel
     HostArray<std::complex<T>, 1> postphase(n);
     std::vector<std::complex<T>> postphase_ref(n);

     if (dim == 1) {
       std::array<const T *, 1> out_loc = {output_loc1.data()};
       std::array<T, 1> out_offset = {1.0};
       std::array<T, 1> in_offset = {-1.0};
       ref::compute_postphase<T, 1>(sign, n, phi_hat.data(), out_loc,in_offset, out_offset,
                                    postphase_ref.data());
       compute_postphase<T, 1>(sign, n, phi_hat.data(), out_loc, in_offset, out_offset,
                               postphase.data());
     }

     if (dim == 2) {
       std::array<const T *, 2> out_loc = {output_loc1.data(),
                                           output_loc2.data()};
       std::array<T, 2> out_offset = {1.0, 2.0};
       std::array<T, 2> in_offset = {-1.0, -2.0};
       ref::compute_postphase<T, 2>(sign, n, phi_hat.data(), out_loc,in_offset, out_offset,
                                    postphase_ref.data());
       compute_postphase<T, 2>(sign, n, phi_hat.data(), out_loc, in_offset, out_offset,
                               postphase.data());
     }

     if (dim == 3) {
       std::array<const T *, 3> out_loc = {
           output_loc1.data(), output_loc2.data(), output_loc3.data()};
       std::array<T, 3> out_offset = {1.0, 2.0, 3.0};
       std::array<T, 3> in_offset = {-1.0, -2.0, -3.0};
       ref::compute_postphase<T, 3>(sign, n, phi_hat.data(), out_loc, in_offset,
                                    out_offset, postphase_ref.data());
       compute_postphase<T, 3>(sign, n, phi_hat.data(), out_loc, in_offset,
                               out_offset, postphase.data());
     }

     for (IntType i = 0; i < n; ++i) {
       EXPECT_NEAR(postphase[i].real(), postphase_ref[i].real(),
                   (T)std::abs(postphase_ref[i].real() * 0.01));
       EXPECT_NEAR(postphase[i].imag(), postphase_ref[i].imag(),
                   (T)std::abs(postphase_ref[i].imag() * 0.01));
     }
   }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PostphaseTest, TestTypes);

TYPED_TEST(PostphaseTest, OneDim13) { this->test_ref(1, 1, 13); }

TYPED_TEST(PostphaseTest, TwoDim24) { this->test_ref(2, -1, 24); }

TYPED_TEST(PostphaseTest, ThreeDim100) { this->test_ref(3, -1, 100); }
