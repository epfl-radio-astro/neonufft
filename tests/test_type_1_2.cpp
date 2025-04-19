#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cmath>

#include "neonufft/config.h"
// ---

#include "neonufft/enums.h"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "neonufft/allocator.hpp"
#include "neonufft/util/math.hpp"
#include "test_thread_pool.hpp"
#include "gtest/gtest.h"
#include "nuft_direct_kernels.hpp"

#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)
#include "neonufft/gpu/memory/copy.hpp"
#include "neonufft/gpu/memory/device_array.hpp"
#include "neonufft/memory/view.hpp"
#include "neonufft/gpu/device_allocator.hpp"
#include "neonufft/gpu/plan.hpp"
#include "neonufft/gpu/types.hpp"
#endif

using namespace neonufft;

namespace {

struct PlanTestParam {
  using TupleType = std::tuple<IntType, std::array<IntType, 3>, double, double, int, bool, int>;
  PlanTestParam(const TupleType& t)
      : num_nu(std::get<0>(t)),
        modes(std::get<1>(t)),
        upsampfac(std::get<2>(t)),
        tol(std::get<3>(t)),
        sign(std::get<4>(t)),
        use_gpu(std::get<5>(t)),
        type(std::get<6>(t)),
        dim(1) {
    if (modes[1]) dim = 2;
    if (modes[2]) dim = 3;
  }

  std::array<IntType, 3> modes;
  IntType num_nu;
  double upsampfac;
  double tol;
  int sign;
  bool use_gpu;
  IntType dim;
  int type;
};

template <typename T, typename GEN>
std::vector<T> rand_vec(GEN&& rand_gen, IntType num, T min, T max) {
  std::uniform_real_distribution<T> dist(min, max);
  std::vector<T> vec(num);
  for (IntType i = 0; i < num; ++i) {
    vec[i] = dist(rand_gen);
  }
  return vec;
}

template <typename T, typename GEN>
std::vector<std::complex<T>> rand_vec_cpx(GEN&& rand_gen, IntType num, T min, T max) {
  std::uniform_real_distribution<T> dist(min, max);
  std::vector<std::complex<T>> vec(num);
  for (IntType i = 0; i < num; ++i) {
    vec[i] = std::complex<T>{dist(rand_gen), dist(rand_gen)};
  }
  return vec;
}

template <typename T>
void compare_output(T element_tolerance, T global_tolerance, T global_percentage_threshold,
                    const std::vector<std::complex<T>>& left,
                    const std::vector<std::complex<T>>& right) {
  ASSERT_EQ(left.size(), right.size());

  IntType count_below_tol = 0;
  for (IntType i = 0; i < left.size(); ++i) {
    T rel_error_real = left[i].real() ? std::abs(right[i].real() - left[i].real()) / left[i].real()
                                      : right[i].real();
    T rel_error_imag = left[i].imag() ? std::abs(right[i].imag() - left[i].imag()) / left[i].imag()
                                      : right[i].imag();

    const auto relative_error = std::norm(right[i] - left[i]) / std::norm(left[i]);

    if (relative_error <= global_tolerance) {
      ++count_below_tol;
    }

    // test each value with larger tolerance, since this is no hard upper
    // bound
    EXPECT_LT(relative_error, element_tolerance);
  }

  EXPECT_GT(double(count_below_tol) / double(left.size()), global_percentage_threshold);
}

template <typename T, IntType DIM>
void test_transform(int type, bool use_gpu, int sign, double upsampfac, double tol,
                    std::array<IntType, DIM> modes, IntType num_nu,
                    std::array<const T*, DIM> points) {
  Options opt;
  opt.upsampfac = upsampfac;
  opt.tol = tol;

  const auto num_modes = std::accumulate(modes.begin(), modes.end(), IntType(1), std::multiplies{});

  std::minstd_rand rand_gen(42);

  std::vector<std::complex<T>> ref_output(type == 2 ? num_nu : num_modes);
  std::vector<std::complex<T>> output(ref_output.size());
  auto input = rand_vec_cpx<T>(rand_gen, type == 2 ? num_modes : num_nu, -4, 4);

  std::array<IntType, DIM> strides;
  strides[0] = 1;
  for (IntType i = 1; i < DIM; ++i) {
    strides[i] = strides[i - 1] * modes[i - 1];
  }

  std::shared_ptr<Allocator> device_alloc;

  if (use_gpu) {
#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)
    device_alloc.reset(new gpu::DeviceAllocator());

    gpu::DeviceArray<gpu::ComplexType<T>, 1> input_device(input.size(), device_alloc);
    gpu::DeviceArray<gpu::ComplexType<T>, 1> output_device(output.size(), device_alloc);
    gpu::memcopy(ConstHostView<std::complex<T>, 1>(input.data(), input.size(), 1), input_device,
                 nullptr);
    gpu::DeviceArray<T, 2> points_device({num_nu, DIM}, device_alloc);
    decltype(points) points_device_pointer;
    for (IntType dim = 0; dim < DIM; ++dim) {
      gpu::memcopy(ConstHostView<T, 1>(points[dim], num_nu, 1), points_device.slice_view(dim),
                   nullptr);
      points_device_pointer[dim] = points_device.slice_view(dim).data();
    }

    gpu::Plan<T, DIM> plan(opt, sign, num_nu, points_device_pointer, modes, nullptr);

    if (type == 2) {
      plan.transform_type_2(input_device.data(), strides, output_device.data());
    } else {
      plan.transform_type_1(input_device.data(), output_device.data(), strides);
    }

    gpu::memcopy(output_device, HostView<std::complex<T>, 1>(output.data(), output.size(), 1),
                 nullptr);
    gpu::api::stream_synchronize(nullptr);
#else
    ASSERT_TRUE(false);
#endif
  } else {
    Plan<T, DIM> plan(opt, sign, num_nu, points, modes);
    if (type == 2) {
      plan.transform_type_2(input.data(), strides, output.data());
    } else {
      plan.transform_type_1(input.data(), output.data(), strides);
    }
  }

  if (type == 2) {
    // nu output
    test::thread_pool.parallel_for({0, IntType(num_nu)}, 1, [&](IntType, BlockRange range) {
      // split output points among threads
      auto local_points = points;
      for (IntType dim = 0; dim < DIM; ++dim) {
        local_points[dim] += range.begin;
      }
      test::nuft_direct_t2<T, DIM>(sign, modes, input.data(), range.end - range.begin, local_points,
                                   ref_output.data() + range.begin, strides);
    });

  } else {
    // mode grid output
    test::nuft_direct_t1<T, DIM>(sign, modes, input.data(), num_nu, points, ref_output.data(),
                                 strides);
  }

  // Tolerance is not an absolute upper bound. Type 1 is far less accurate in single precision.
  T element_tolerance = 400 * tol;
  if (type == 1 && std::is_same_v<float, T>) {
    element_tolerance = 20000 * tol;
  }

  // we expect 85% to be below tolerance for large output numbers
  compare_output<T>(element_tolerance, tol, 0.85, ref_output, output);
}

}  // namespace

template <typename T>
class PlanTest : public ::testing::TestWithParam<PlanTestParam::TupleType> {
public:
  void test_random() {
    PlanTestParam param(GetParam());
    std::minstd_rand rand_gen(42);

    auto points_x = rand_vec<T>(rand_gen, param.num_nu, 20, 30);

    if (param.dim == 1) {
      test_transform<T, 1>(param.type, param.use_gpu, param.sign, param.upsampfac, param.tol,
                           {param.modes[0]}, param.num_nu, {points_x.data()});
    } else {
      auto points_y = rand_vec<T>(rand_gen, param.num_nu, -20, 20);

      if (param.dim == 2) {
        test_transform<T, 2>(param.type, param.use_gpu, param.sign, param.upsampfac, param.tol,
                             {param.modes[0], param.modes[1]}, param.num_nu,
                             {points_x.data(), points_y.data()});
      } else {
        auto points_z = rand_vec<T>(rand_gen, param.num_nu, -20, -15);
        test_transform<T, 3>(param.type, param.use_gpu, param.sign, param.upsampfac, param.tol,
                             {param.modes[0], param.modes[1], param.modes[2]}, param.num_nu,
                             {points_x.data(), points_y.data(), points_z.data()});
      }
    }
  }
};

using PlanTestFloat = PlanTest<float>;

TEST_P(PlanTestFloat, transform) { test_random(); }

using PlanTestDouble = PlanTest<double>;

TEST_P(PlanTestDouble, transform) { test_random(); }

// TEST(PlanTestEdges, d1) {
//   std::array<double, 2> points_x = {-math::pi<double>, math::pi<double>};
//   // test_transform<double>(-1, 2, 1e-15, points_x.size(), points_x.data(), nullptr,
//   //                    nullptr, points_x.size(), points_x.data(), nullptr,
//   //                    nullptr);
// }

// TEST(PlanTestEdges, d2) {
//   std::array<double, 3> points = {-math::pi<double>, 0.0, math::pi<double>};
//   std::vector<double> points_x;
//   std::vector<double> points_y;
//   for(const auto& x : points) {
//     for (const auto &y : points) {
//       points_x.emplace_back(x);
//       points_y.emplace_back(y);
//     }
//   }
//   // test_transform<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
//   //                    points_y.data(), nullptr, points_x.size(), points_x.data(),
//   //                    points_y.data(), nullptr);
// }

// TEST(PlanTestEdges, d3) {
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
//   // test_transform<double>(-1, 2, 1e-15, points_x.size(), points_x.data(),
//   //                    points_y.data(), points_z.data(), points_x.size(),
//   //                    points_x.data(), points_y.data(), points_z.data());
// }

static auto param_type_names(const ::testing::TestParamInfo<PlanTestParam::TupleType>& info)
    -> std::string {
  std::stringstream stream;

  PlanTestParam param(info.param);

  if (param.use_gpu)
    stream << "gpu_";
  else
    stream << "host_";
  stream << "type_" << param.type;
  stream << "_d_" << param.dim;
  stream << "_n_" << param.num_nu;
  stream << "_modes_" << param.modes[0];
  stream << "_" << param.modes[1];
  stream << "_" << param.modes[2];
  const int upsampfacInt = param.upsampfac;
  const int upsampfacFrac = int(param.upsampfac * 100) % 100;
  stream << "_up_" << upsampfacInt << "_" << upsampfacFrac;
  stream << "_tol_" << int(-std::log10(param.tol));
  stream << "_sign_" << (param.sign < 0 ? "minus" : "plus");

  return stream.str();
}

static auto generate_modes(std::vector<IntType> sizes) -> std::vector<std::array<IntType, 3>> {
  std::vector<std::array<IntType, 3>> modes;

  // 1D
  for (const auto& s1 : sizes) {
    modes.emplace_back(std::array<IntType, 3>{s1, 0, 0});
  }

  // 2D
  for (const auto& s1 : sizes) {
    for (const auto& s2 : sizes) {
      modes.emplace_back(std::array<IntType, 3>{s1, s2, 0});
    }
  }

  // 3D
  for (const auto& s1 : sizes) {
    for (const auto& s2 : sizes) {
      for (const auto& s3 : sizes) {
        modes.emplace_back(std::array<IntType, 3>{s1, s2, s3});
      }
    }
  }

  return modes;
};

#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)
#define NEONUFFT_PU_VALUES false, true
#else
#define NEONUFFT_PU_VALUES false
#endif

INSTANTIATE_TEST_SUITE_P(
    Modes, PlanTestFloat,
    ::testing::Combine(::testing::Values<IntType>(100),                       // number of in points
                       ::testing::ValuesIn(generate_modes({1, 2, 23, 256})),  // modes
                       ::testing::Values<double>(2.0),                        // upsampling factor
                       ::testing::Values<double>(1e-5),                       // tolerance
                       ::testing::Values<int>(-1),                            // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES),           // cpu, gpu
                       ::testing::Values<int>(1, 2)),  // transform type 1 or 2
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Sign, PlanTestFloat,
    ::testing::Combine(::testing::Values<IntType>(10),               // number of in points
                       ::testing::ValuesIn(generate_modes({16})),    // modes
                       ::testing::Values<double>(2.0),               // upsampling factor
                       ::testing::Values<double>(1e-5),              // tolerance
                       ::testing::Values<int>(1, -1),                // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES),  // cpu, gpu
                       ::testing::Values<int>(1, 2)),                // transform type 1 or 2
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Up, PlanTestFloat,
    ::testing::Combine(::testing::Values<IntType>(10),               // number of in points
                       ::testing::ValuesIn(generate_modes({16})),    // modes
                       ::testing::Values<double>(1.25, 1.6, 2.0),    // upsampling factor
                       ::testing::Values<double>(1e-5),              // tolerance
                       ::testing::Values<int>(-1),                   // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES),  // cpu, gpu
                       ::testing::Values<int>(1, 2)),                // transform type 1 or 2
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Up, PlanTestDouble,
    ::testing::Combine(::testing::Values<IntType>(10),               // number of in points
                       ::testing::ValuesIn(generate_modes({32})),    // modes
                       ::testing::Values<double>(1.25, 1.6, 2.0),    // upsampling factor
                       ::testing::Values<double>(1e-5),              // tolerance
                       ::testing::Values<int>(-1),                   // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES),  // cpu, gpu
                       ::testing::Values<int>(1, 2)),                // transform type 1 or 2
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Tol, PlanTestDouble,
    ::testing::Combine(::testing::Values<IntType>(20),               // number of in points
                       ::testing::ValuesIn(generate_modes({40})),     // modes
                       ::testing::Values<double>(2.0),                // upsampling factor
                       ::testing::Values<double>(1e-3, 1e-8, 1e-14),  // tolerance
                       ::testing::Values<int>(-1),                    // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES),   // cpu, gpu
                       ::testing::Values<int>(1, 2)),                 // transform type 1 or 2
    param_type_names);
