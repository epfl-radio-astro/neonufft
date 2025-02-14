#include <algorithm>
#include <array>
#include <complex>
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

struct Type3TestParam {
 using TupleType = std::tuple<IntType, IntType, IntType, double, double, int>;
 Type3TestParam(const TupleType &t)
     : dim(std::get<0>(t)), num_in(std::get<1>(t)), num_out(std::get<2>(t)),
       upsampfac(std::get<3>(t)), tol(std::get<4>(t)), sign(std::get<5>(t)) {}

 IntType dim;
 IntType num_in;
 IntType num_out;
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
template <typename T, IntType DIM>
void transform_t3(int sign, IntType num_in,
                  std::array<const T *, DIM> in_points,
                  const std::complex<T> *in, IntType num_out,
                  std::array<const T *, DIM> out_points, std::complex<T> *out) {

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    std::complex<T> sum = {0, 0};
    for (IntType idx_in = 0; idx_in < num_in; ++idx_in) {
      T dot = in_points[0][idx_in] * out_points[0][idx_out];
      for (IntType dim = 1; dim < DIM; ++dim) {
        dot += in_points[dim][idx_in] * out_points[dim][idx_out];
      }
      sum += in[idx_in] * std::exp(std::complex<T>{0, sign * dot});
    }
    out[idx_out] = sum;
  }
}
} // namespace ref

template <typename T>
void compare_t3(int sign, double upsampfac, double tol, IntType num_in,
                const T *in_points_x, const T *in_points_y,
                const T *in_points_z, IntType num_out, const T *out_points_x,
                const T *out_points_y, const T *out_points_z) {

  Options opt;
  opt.upsampfac = upsampfac;
  opt.tol = tol;

  std::minstd_rand rand_gen(42);

  auto input = rand_vec_cpx<T>(rand_gen, num_in, -4, 4);

  std::vector<std::complex<T>> output(num_out);
  std::vector<std::complex<T>> output_ref(num_out);

  if (!in_points_y) {
    // DIM == 1
    PlanT3<T, 1> plan(opt, sign, num_in, {in_points_x}, num_out,
                      {out_points_x});
    plan.add_input(input.data());
    plan.transform(output.data());

    ref::transform_t3<T, 1>(sign, num_in, {in_points_x}, input.data(),
                            num_out, {out_points_x}, output_ref.data());

  } else if (!in_points_z) {
    // DIM == 2

    PlanT3<T, 2> plan(opt, sign, num_in, {in_points_x, in_points_y},
                      num_out, {out_points_x, out_points_y});
    plan.add_input(input.data());
    plan.transform(output.data());

    ref::transform_t3<T, 2>(sign, num_in, {in_points_x, in_points_y},
                            input.data(), num_out,
                            {out_points_x, out_points_y}, output_ref.data());
  } else {
    // DIM == 3

    PlanT3<T, 3> plan(opt, sign, num_in,
                      {in_points_x, in_points_y, in_points_z}, num_out,
                      {out_points_x, out_points_y, out_points_z});
    plan.add_input(input.data());
    plan.transform(output.data());

    ref::transform_t3<T, 3>(
        sign, num_in, {in_points_x, in_points_y, in_points_z},
        input.data(), num_out, {out_points_x, out_points_y, out_points_z},
        output_ref.data());
  }

  IntType count_below_tol = 0;
  for (IntType i = 0; i < num_out; ++i) {
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
    EXPECT_LT(relative_error, 300 * tol);
  }

  // we expect 94% to be below tolerance for large output numbers
  const double expected = num_out > 20 ? 0.94 : 0.85;
  EXPECT_GT(double(count_below_tol) / double(num_out), expected);
}

} // namespace

template <typename T>
class Type3Test
    : public ::testing::TestWithParam<Type3TestParam::TupleType> {
public:
 void test_random() {
   Type3TestParam param(GetParam());
   std::minstd_rand rand_gen(42);

   auto in_points_x = rand_vec<T>(rand_gen, param.num_in, 20, 30);
   auto out_points_x = rand_vec<T>(rand_gen, param.num_out, 20, 30);
   if (param.dim == 1) {
     compare_t3<T>(param.sign, param.upsampfac, param.tol, param.num_in,
                in_points_x.data(), nullptr, nullptr, param.num_out,
                out_points_x.data(), nullptr, nullptr);
   } else {

     auto in_points_y = rand_vec<T>(rand_gen, param.num_in, -20, 20);
     auto out_points_y = rand_vec<T>(rand_gen, param.num_out, -20, 20);

     if (param.dim == 2) {
       compare_t3<T>(param.sign, param.upsampfac, param.tol, param.num_in,
               in_points_x.data(), in_points_y.data(), nullptr, param.num_out,
               out_points_x.data(), out_points_y.data(), nullptr);
     } else {
       auto in_points_z = rand_vec<T>(rand_gen, param.num_in, -20, -15);
       auto out_points_z = rand_vec<T>(rand_gen, param.num_out, -20, -15);
       compare_t3<T>(param.sign, param.upsampfac, param.tol, param.num_in,
                     in_points_x.data(), in_points_y.data(), in_points_z.data(),
                     param.num_out, out_points_x.data(), out_points_y.data(),
                     out_points_z.data());
     }
   }
 }

};

using Type3TestFloat = Type3Test<float>;

TEST_P(Type3TestFloat, transform) { test_random(); }

using Type3TestDouble = Type3Test<double>;

TEST_P(Type3TestDouble, transform) { test_random(); }

TEST(Type3TestEdges, d1) {
  std::array<double, 2> points_x = {-math::pi<double>, math::pi<double>};
  compare_t3<double>(-1, 2, 1e-15, points_x.size(), points_x.data(), nullptr,
                     nullptr, points_x.size(), points_x.data(), nullptr,
                     nullptr);
}

TEST(Type3TestEdges, d2) {
  std::array<double, 3> points = {-math::pi<double>, 0.0, math::pi<double>};
  std::vector<double> points_x;
  std::vector<double> points_y;
  for(const auto& x : points) {
    for (const auto &y : points) {
      points_x.emplace_back(x);
      points_y.emplace_back(y);
    }
  }
  compare_t3<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
                     points_y.data(), nullptr, points_x.size(), points_x.data(),
                     points_y.data(), nullptr);
}

TEST(Type3TestEdges, d3) {
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
  compare_t3<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
                     points_y.data(), points_z.data(), points_x.size(),
                     points_x.data(), points_y.data(), points_z.data());
}


static auto param_type_names(
    const ::testing::TestParamInfo<Type3TestParam::TupleType> &info)
    -> std::string {
  std::stringstream stream;

  Type3TestParam param(info.param);

  stream << "d_" << param.dim;
  stream << "_m_" << param.num_in;
  stream << "_n_" << param.num_out;
  stream << "_up_" << param.upsampfac;
  stream << "_tol_" << int(-std::log10(param.tol));
  stream << "_sign_" << (param.sign < 0 ? "minus" : "plus");

  return stream.str();
}

INSTANTIATE_TEST_SUITE_P(
    Type3, Type3TestFloat,
    ::testing::Combine(
        ::testing::Values<IntType>(1, 2),            // dimension
        ::testing::Values<IntType>(1, 10, 200, 503), // number of in points
        ::testing::Values<IntType>(1, 10, 200, 503), // number of out points
        ::testing::Values<double>(2.0),              // upsampling factor
        ::testing::Values<double>(1e-4, 1e-7),       // tolerance
        ::testing::Values<int>(1, -1)),              // sign
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type3, Type3TestDouble,
    ::testing::Combine(
        ::testing::Values<IntType>(1, 2),            // dimension
        ::testing::Values<IntType>(1, 10, 200, 503), // number of in points
        ::testing::Values<IntType>(1, 10, 200, 503), // number of out points
        ::testing::Values<double>(2.0),              // upsampling factor
        ::testing::Values<double>(1e-4, 1e-7),       // tolerance
        ::testing::Values<int>(1, -1)),              // sign
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type3D3, Type3TestFloat,
    ::testing::Combine(
        ::testing::Values<IntType>(3),         // dimension
        ::testing::Values<IntType>(1, 10, 99), // number of in points
        ::testing::Values<IntType>(1, 10, 99), // number of out points
        ::testing::Values<double>(2.0),        // upsampling factor
        ::testing::Values<double>(1e-4, 1e-7), // tolerance
        ::testing::Values<int>(1)),            // sign
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type3D3, Type3TestDouble,
    ::testing::Combine(
        ::testing::Values<IntType>(3),         // dimension
        ::testing::Values<IntType>(1, 10, 99), // number of in points
        ::testing::Values<IntType>(1, 10, 99), // number of out points
        ::testing::Values<double>(2.0),        // upsampling factor
        ::testing::Values<double>(1e-4, 1e-7), // tolerance
        ::testing::Values<int>(1)),            // sign
    param_type_names);
