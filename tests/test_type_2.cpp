#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <cmath>

#include "gpu/util/runtime_api.hpp"
#include "neonufft/config.h"
// ---

#include "neonufft/enums.h"
#include "neonufft/gpu/types.hpp"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "util/math.hpp"
#include "gtest/gtest.h"

#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)
#include "gpu/memory/copy.hpp"
#include "gpu/memory/device_array.hpp"
#include "memory/view.hpp"
#include "neonufft/gpu/device_allocator.hpp"
#include "neonufft/gpu/plan.hpp"
#endif

using namespace neonufft;

namespace {

struct Type2TestParam {
  using TupleType = std::tuple<IntType, IntType, IntType, IntType,
                               double, double, int, bool>;
  Type2TestParam(const TupleType& t)
      : num_nu(std::get<0>(t)),
        modes({std::get<1>(t), std::get<2>(t), std::get<3>(t)}),
        upsampfac(std::get<4>(t)),
        tol(std::get<5>(t)),
        sign(std::get<6>(t)),
        use_gpu(std::get<7>(t)),
        dim(1) {
    if (modes[1] > 1 && modes[2] > 1)
      dim = 3;
    else if (modes[1] > 1)
      dim = 2;
  }

  std::array<IntType, 3> modes;
  IntType num_nu;
  double upsampfac;
  double tol;
  int sign;
  bool use_gpu;
  IntType dim;
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
void transform_t2(int sign, std::array<IntType, 1> modes,
                  const std::complex<T> *in, IntType num_out,
                  std::array<const T *, 1> out_points, std::complex<T> *out,
                  std::array<IntType, 1>) {

  const IntType k_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    std::complex<T> sum = {0, 0};
    for (IntType k0 = 0; k0 < modes[0]; ++k0) {

      T dot = (k0 - k_offset) * out_points[0][idx_out];

      sum += in[k0] * std::exp(std::complex<T>{0, sign * dot});
    }
    out[idx_out] = sum;
  }
}

// 2D
template <typename T>
void transform_t2(int sign, std::array<IntType, 2> modes,
                  const std::complex<T> *in, IntType num_out,
                  std::array<const T *, 2> out_points, std::complex<T> *out,
                  std::array<IntType, 2> strides) {

  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    std::complex<T> sum = {0, 0};
    for (IntType k1 = 0; k1 < modes[1]; ++k1) {
      for (IntType k0 = 0; k0 < modes[0]; ++k0) {

        T dot = (k1 - k1_offset) * out_points[1][idx_out] +
                (k0 - k0_offset) * out_points[0][idx_out];

        //TODO: change in index
        sum +=
            in[k0 + k1 * strides[1]] * std::exp(std::complex<T>{0, sign * dot});
      }
    }
    out[idx_out] = sum;
  }
}

// 3D
template <typename T>
void transform_t2(int sign, std::array<IntType, 3> modes,
                  const std::complex<T> *in, IntType num_out,
                  std::array<const T *, 3> out_points, std::complex<T> *out,
                  std::array<IntType, 3> strides) {

  const IntType k0_offset = modes[0] % 2 ? (modes[0] - 1) / 2 : modes[0] / 2;
  const IntType k1_offset = modes[1] % 2 ? (modes[1] - 1) / 2 : modes[1] / 2;
  const IntType k2_offset = modes[2] % 2 ? (modes[2] - 1) / 2 : modes[2] / 2;

  for (IntType idx_out = 0; idx_out < num_out; ++idx_out) {
    std::complex<T> sum = {0, 0};
    for (IntType k2 = 0; k2 < modes[2]; ++k2) {
      for (IntType k1 = 0; k1 < modes[1]; ++k1) {
        for (IntType k0 = 0; k0 < modes[0]; ++k0) {

          T dot = (k2 - k2_offset) * out_points[2][idx_out] +
                  (k1 - k1_offset) * out_points[1][idx_out] +
                  (k0 - k0_offset) * out_points[0][idx_out];

          // TODO: change in index
          sum += in[k0 + k1 * strides[1] + k2 * strides[2]] *
                 std::exp(std::complex<T>{0, sign * dot});
        }
      }
    }
    out[idx_out] = sum;
  }
}

} // namespace ref

template <typename T, IntType DIM>
void compare_t2(bool use_gpu, int sign, double upsampfac, double tol,
                std::array<IntType, DIM> modes, IntType num_nu,
                std::array<const T*, DIM> out_points) {
  Options opt;
  opt.upsampfac = upsampfac;
  opt.tol = tol;

  std::minstd_rand rand_gen(42);

  auto input = rand_vec_cpx<T>(rand_gen,
                               std::accumulate(modes.begin(), modes.end(),
                                               IntType(1), std::multiplies{}),
                               -4, 4);

  std::vector<std::complex<T>> output(num_nu);
  std::vector<std::complex<T>> output_ref(num_nu);

  std::array<IntType, DIM> strides;
  strides[0] = 1;
  for(IntType i = 1; i < DIM; ++i) {
   strides[i] = strides[i - 1] * modes[i - 1];
  }

  std::shared_ptr<Allocator> device_alloc;

  if (use_gpu) {
#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)
   device_alloc.reset(new gpu::DeviceAllocator());

   gpu::DeviceArray<gpu::ComplexType<T>, 1> input_device(input.size(), device_alloc);
   gpu::DeviceArray<gpu::ComplexType<T>, 1> output_device(output.size(), device_alloc);
   gpu::memcopy(ConstHostView<std::complex<T>,1>(input.data(), input.size(), 1), input_device, nullptr);

   gpu::Plan<T, DIM> plan(opt, sign, num_nu, out_points, modes, nullptr);
   plan.transform_type_2(input_device.data(), strides, output_device.data());

   gpu::memcopy(output_device, HostView<std::complex<T>, 1>(output.data(), output.size(), 1),
                nullptr);
   gpu::api::stream_synchronize(nullptr);
#else
   ASSERT_TRUE(false);
#endif
  } else {
    Plan<T, DIM> plan(opt, sign, num_nu, out_points, modes);
    plan.transform_type_2(input.data(), strides, output.data());
  }

  ref::transform_t2<T>(sign, modes, input.data(), num_nu, out_points,
                       output_ref.data(), strides);

  IntType count_below_tol = 0;
  for (IntType i = 0; i < num_nu; ++i) {
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
    EXPECT_LT(relative_error, 400 * tol);
  }

  // we expect 85% to be below tolerance for large output numbers
  // const double expected = num_nu > 20 ? 0.94 : 0.85;
  const double expected = 0.85;
  EXPECT_GT(double(count_below_tol) / double(num_nu), expected);
}

} // namespace

template <typename T>
class Type2Test
    : public ::testing::TestWithParam<Type2TestParam::TupleType> {
public:
 void test_random() {
   Type2TestParam param(GetParam());
   std::minstd_rand rand_gen(42);

   auto out_points_x = rand_vec<T>(rand_gen, param.num_nu, 20, 30);

   if (param.dim == 1) {
      compare_t2<T, 1>(param.use_gpu, param.sign, param.upsampfac, param.tol, {param.modes[0]},
                       param.num_nu, {out_points_x.data()});
   } else {

     auto out_points_y = rand_vec<T>(rand_gen, param.num_nu, -20, 20);

     if (param.dim == 2) {
       compare_t2<T, 2>(param.use_gpu, param.sign, param.upsampfac, param.tol,
                        {param.modes[0], param.modes[1]}, param.num_nu,
                        {out_points_x.data(), out_points_y.data()});
     } else {
       auto out_points_z = rand_vec<T>(rand_gen, param.num_nu, -20, -15);
       compare_t2<T, 3>(param.use_gpu, param.sign, param.upsampfac, param.tol,
                        {param.modes[0], param.modes[1], param.modes[2]}, param.num_nu,
                        {out_points_x.data(), out_points_y.data(), out_points_z.data()});
     }
   }
 }

};

using Type2TestFloat = Type2Test<float>;

TEST_P(Type2TestFloat, transform) { test_random(); }

using Type2TestDouble = Type2Test<double>;

TEST_P(Type2TestDouble, transform) { test_random(); }

// TEST(Type2TestEdges, d1) {
//   std::array<double, 2> points_x = {-math::pi<double>, math::pi<double>};
//   // compare_t2<double>(-1, 2, 1e-15, points_x.size(), points_x.data(), nullptr,
//   //                    nullptr, points_x.size(), points_x.data(), nullptr,
//   //                    nullptr);
// }

// TEST(Type2TestEdges, d2) {
//   std::array<double, 3> points = {-math::pi<double>, 0.0, math::pi<double>};
//   std::vector<double> points_x;
//   std::vector<double> points_y;
//   for(const auto& x : points) {
//     for (const auto &y : points) {
//       points_x.emplace_back(x);
//       points_y.emplace_back(y);
//     }
//   }
//   // compare_t2<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
//   //                    points_y.data(), nullptr, points_x.size(), points_x.data(),
//   //                    points_y.data(), nullptr);
// }

// TEST(Type2TestEdges, d3) {
//   std::array<double, 3> points = {-math::pi<double>, 0.0, math::pi<double>};
//   std::vector<double> points_x;
//   std::vector<double> points_y;
//   std::vector<double> points_z;
//   for (const auto &x : points) {
//     for (const auto &y : points) {
//       for (const auto &z : points) {
//         points_x.emplace_back(x);
//         points_y.emplace_back(y);
//         points_z.emplace_back(z);
//       }
//     }
//   }
//   // compare_t2<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
//   //                    points_y.data(), points_z.data(), points_x.size(),
//   //                    points_x.data(), points_y.data(), points_z.data());
// }

static auto param_type_names(
    const ::testing::TestParamInfo<Type2TestParam::TupleType> &info)
    -> std::string {
  std::stringstream stream;

  Type2TestParam param(info.param);

  if (param.use_gpu)
    stream << "gpu_";
  else
    stream << "host_";
  stream << "d_" << param.dim;
  stream << "_n_" << param.num_nu;
  stream << "_modes_" << param.modes[0];
  stream << "_" << param.modes[1];
  stream << "_" << param.modes[2];
  stream << "_up_" << param.upsampfac;
  stream << "_tol_" << int(-std::log10(param.tol));
  stream << "_sign_" << (param.sign < 0 ? "minus" : "plus");

  return stream.str();
}

#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)
#define NEONUFFT_PU_VALUES false, true
#else
#define NEONUFFT_PU_VALUES false
#endif

INSTANTIATE_TEST_SUITE_P(
    Type2, Type2TestFloat,
    ::testing::Combine(::testing::Values<IntType>(1, 10, 200,
                                                  503),               // number of in points
                       ::testing::Values<IntType>(1, 10, 200, 203),   // mode x
                       ::testing::Values<IntType>(1, 49),             // mode y
                       ::testing::Values<IntType>(1, 49),             // mode z
                       ::testing::Values<double>(2.0),                // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-6),         // tolerance
                       ::testing::Values<int>(1, -1),                 // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES)),  // cpu, gpu
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type2, Type2TestDouble,
    ::testing::Combine(::testing::Values<IntType>(1, 10, 200,
                                                  503),               // number of in points
                       ::testing::Values<IntType>(1, 10, 100, 203),   // mode x
                       ::testing::Values<IntType>(1, 49),             // mode y
                       ::testing::Values<IntType>(1, 30),             // mode z
                       ::testing::Values<double>(2.0),                // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-7),         // tolerance
                       ::testing::Values<int>(1, -1),                 // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES)),  // cpu, gpu
    param_type_names);
