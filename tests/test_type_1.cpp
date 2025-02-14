#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <cmath>

#include "neonufft/config.h"

#include "neonufft/enums.h"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "util/math.hpp"
#include "gtest/gtest.h"

using namespace neonufft;

namespace {

struct Type1TestParam {
  using TupleType = std::tuple<IntType, IntType, IntType, IntType, IntType,
                               double, double, int>;
  Type1TestParam(const TupleType &t)
      : dim(std::get<0>(t)), num_nu(std::get<1>(t)),
        modes({std::get<2>(t), std::get<3>(t), std::get<4>(t)}),
        upsampfac(std::get<5>(t)), tol(std::get<6>(t)), sign(std::get<7>(t)) {}

  IntType dim;
  std::array<IntType, 3> modes;
  IntType num_nu;
  double upsampfac;
  double tol;
  int sign;
};

template <typename T, typename GEN>
std::vector<T> rand_vec(GEN &&rand_gen, IntType num, T min, T max) {
  std::uniform_real_distribution<T> dist(min, max);
  std::vector<T> vec(num);
  for(IntType i =0; i<num; ++i) {
   vec[i] = dist(rand_gen);
  }
  return vec;
}

template <typename T, typename GEN>
std::vector<std::complex<T>> rand_vec_cpx(GEN &&rand_gen, IntType num, T min, T max) {
  std::uniform_real_distribution<T> dist(min, max);
  std::vector<std::complex<T>> vec(num);
  for (IntType i = 0; i < num; ++i) {
    vec[i] = std::complex<T>{dist(rand_gen), dist(rand_gen)};
  }
  return vec;
}

namespace ref {

// 1D
template <typename T>
void transform_t1(int sign, std::array<IntType, 1> modes,
                  const std::complex<T> *in, IntType num_in,
                  std::array<const T *, 1> in_points, std::complex<T> *out,
                  std::array<IntType, 1>) {

  const IntType k_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;

  for (IntType k0 = 0; k0 < modes[0]; ++k0) {
    std::complex<T> sum = {0, 0};
    for (IntType idx_in = 0; idx_in < num_in; ++idx_in) {

      T dot = (k0 - k_offset) * in_points[0][idx_in];

      sum += in[idx_in] * std::exp(std::complex<T>{0, sign * dot});
    }
    out[k0] = sum;
  }
}

// 2D
template <typename T>
void transform_t1(int sign, std::array<IntType, 2> modes,
                  const std::complex<T> *in, IntType num_in,
                  std::array<const T *, 2> in_points, std::complex<T> *out,
                  std::array<IntType, 2> strides) {

  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;

  for (IntType k1 = 0; k1 < modes[1]; ++k1) {
    for (IntType k0 = 0; k0 < modes[0]; ++k0) {
      std::complex<T> sum = {0, 0};
      for (IntType idx_in = 0; idx_in < num_in; ++idx_in) {

        T dot = (k0 - k0_offset) * in_points[0][idx_in] +
                (k1 - k1_offset) * in_points[1][idx_in];

        sum += in[idx_in] * std::exp(std::complex<T>{0, sign * dot});
      }
      out[k0 + k1 * strides[1]] = sum;
    }
  }
}

// 3D
template <typename T>
void transform_t1(int sign, std::array<IntType, 3> modes,
                  const std::complex<T> *in, IntType num_in,
                  std::array<const T *, 3> in_points, std::complex<T> *out,
                  std::array<IntType, 3> strides) {

  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;
  const IntType k2_offset = modes[2] % 2 ? (modes[2] - 1) / 2 : modes[2] / 2;

  for (IntType k2 = 0; k2 < modes[2]; ++k2) {
    for (IntType k1 = 0; k1 < modes[1]; ++k1) {
      for (IntType k0 = 0; k0 < modes[0]; ++k0) {
        std::complex<T> sum = {0, 0};
        for (IntType idx_in = 0; idx_in < num_in; ++idx_in) {

          T dot = (k0 - k0_offset) * in_points[0][idx_in] +
                  (k1 - k1_offset) * in_points[1][idx_in] +
                  (k2 - k2_offset) * in_points[2][idx_in];

          sum += in[idx_in] * std::exp(std::complex<T>{0, sign * dot});
        }
        out[k0 + k1 * strides[1]] = sum;
      }
    }
  }
}

} // namespace ref

template <typename T, IntType DIM>
void compare_t1(int sign, double upsampfac, double tol,
                std::array<IntType, DIM> modes, IntType num_nu,
                std::array<const T *, DIM> out_points) {

  Options opt;
  opt.upsampfac = upsampfac;
  opt.tol = tol;

  std::minstd_rand rand_gen(42);

  std::vector<std::complex<T>> output(std::accumulate(
      modes.begin(), modes.end(), IntType(1), std::multiplies{}));
  std::vector<std::complex<T>> output_ref(output.size());

  auto input = rand_vec_cpx<T>(rand_gen, num_nu, -3, 3);

  std::array<IntType, DIM> strides;
  strides[0] = 1;
  for(IntType i = 1; i < DIM; ++i) {
   strides[i] = strides[i - 1] * modes[i - 1];
  }

  Plan<T, DIM> plan(opt, sign, num_nu, out_points, modes);

  plan.transform_type_1(input.data(), output.data(), strides);

  ref::transform_t1<T>(sign, modes, input.data(), num_nu, out_points,
                       output_ref.data(), strides);

  IntType count_below_tol = 0;
  for (IntType i = 0; i < output.size(); ++i) {
    T rel_error_real = output_ref[i].real()
                           ? std::abs(output[i].real() - output_ref[i].real()) /
                                 output_ref[i].real()
                           : output[i].real();
    T rel_error_imag = output_ref[i].imag()
                           ? std::abs(output[i].imag() - output_ref[i].imag()) /
                                 output_ref[i].imag()
                           : output[i].imag();

    const auto relative_error =
        std::norm(output[i] - output_ref[i]) / std::norm(output_ref[i]);

    if (relative_error <= tol) {
      ++count_below_tol;
    }

    // test each value with larger tolerance, since this is no hard upper
    // bound
    ASSERT_LT(relative_error, 400 * tol);
  }

  // we expect 85% to be below tolerance for large output numbers
  // const double expected = num_nu > 20 ? 0.94 : 0.85;
  const double expected = 0.85;
  EXPECT_GT(double(count_below_tol) / double(output.size()), expected);
}

} // namespace

template <typename T>
class Type1Test
    : public ::testing::TestWithParam<Type1TestParam::TupleType> {
public:
 void test_random() {
   Type1TestParam param(GetParam());
   std::minstd_rand rand_gen(42);

   auto out_points_x = rand_vec<T>(rand_gen, param.num_nu, 20, 30);

   if (param.dim == 1) {
      compare_t1<T, 1>(param.sign, param.upsampfac, param.tol, {param.modes[0]},
                       param.num_nu, {out_points_x.data()});
   } else {

     auto out_points_y = rand_vec<T>(rand_gen, param.num_nu, -20, 20);

     if (param.dim == 2) {
       compare_t1<T, 2>(param.sign, param.upsampfac, param.tol,
                        {param.modes[0], param.modes[1]}, param.num_nu,
                        {out_points_x.data(), out_points_y.data()});
     } else {
       auto out_points_z = rand_vec<T>(rand_gen, param.num_nu, -20, -15);
       compare_t1<T, 3>(
           param.sign, param.upsampfac, param.tol,
           {param.modes[0], param.modes[1], param.modes[2]}, param.num_nu,
           {out_points_x.data(), out_points_y.data(), out_points_z.data()});
     }
   }
 }

};

using Type1TestFloat = Type1Test<float>;

TEST_P(Type1TestFloat, transform) { test_random(); }

using Type1TestDouble = Type1Test<double>;

TEST_P(Type1TestDouble, transform) { test_random(); }

TEST(Type1TestEdges, d1) {
  std::array<double, 2> points_x = {-math::pi<double>, math::pi<double>};
  // compare_t1<double>(-1, 2, 1e-15, points_x.size(), points_x.data(), nullptr,
  //                    nullptr, points_x.size(), points_x.data(), nullptr,
  //                    nullptr);
}

TEST(Type1TestEdges, d2) {
  std::array<double, 3> points = {-math::pi<double>, 0.0, math::pi<double>};
  std::vector<double> points_x;
  std::vector<double> points_y;
  for(const auto& x : points) {
    for (const auto &y : points) {
      points_x.emplace_back(x);
      points_y.emplace_back(y);
    }
  }
  // compare_t1<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
  //                    points_y.data(), nullptr, points_x.size(), points_x.data(),
  //                    points_y.data(), nullptr);
}

TEST(Type1TestEdges, d3) {
  std::array<double, 3> points = {-math::pi<double>, 0.0, math::pi<double>};
  std::vector<double> points_x;
  std::vector<double> points_y;
  std::vector<double> points_z;
  for (const auto &x : points) {
    for (const auto &y : points) {
      for (const auto &z : points) {
        points_x.emplace_back(x);
        points_y.emplace_back(y);
        points_z.emplace_back(z);
      }
    }
  }
  // compare_t1<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
  //                    points_y.data(), points_z.data(), points_x.size(),
  //                    points_x.data(), points_y.data(), points_z.data());
}

static auto param_type_names(
    const ::testing::TestParamInfo<Type1TestParam::TupleType> &info)
    -> std::string {
  std::stringstream stream;

  Type1TestParam param(info.param);

  stream << "d_" << param.dim;
  stream << "_n_" << param.num_nu;
  stream << "_modes_" << param.modes[0];
  if (param.dim > 1)
    stream << "_" << param.modes[1];
  if (param.dim > 2)
    stream << "_" << param.modes[2];
  stream << "_up_" << param.upsampfac;
  stream << "_tol_" << int(-std::log10(param.tol));
  stream << "_sign_" << (param.sign < 0 ? "minus" : "plus");

  return stream.str();
}

INSTANTIATE_TEST_SUITE_P(
    Type1, Type1TestFloat,
    ::testing::Combine(::testing::Values<IntType>(1, 2, 3), // dimension
                       ::testing::Values<IntType>(1, 10, 200,
                                                  503), // number of in points
                       ::testing::Values<IntType>(1, 10, 200, 503), // mode x
                       ::testing::Values<IntType>(1),               // mode y
                       ::testing::Values<IntType>(1),               // mode z
                       ::testing::Values<double>(2.0), // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-6), // tolerance
                       ::testing::Values<int>(1, -1)),        // sign
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type1, Type1TestDouble,
    ::testing::Combine(::testing::Values<IntType>(1, 2, 3), // dimension
                       ::testing::Values<IntType>(1, 10, 200,
                                                  503), // number of in points
                       ::testing::Values<IntType>(1, 10, 200, 503), // mode x
                       ::testing::Values<IntType>(1),               // mode y
                       ::testing::Values<IntType>(1),               // mode z
                       ::testing::Values<double>(2.0), // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-7), // tolerance
                       ::testing::Values<int>(1, -1)),        // sign
    param_type_names);
