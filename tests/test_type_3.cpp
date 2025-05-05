#include <algorithm>
#include <array>
#include <complex>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cmath>

#include "neonufft/config.h"
//---

#include "gtest/gtest.h"
#include "neonufft/enums.h"
#include "neonufft/plan.hpp"
#include "neonufft/types.hpp"
#include "neonufft/util/math.hpp"
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

struct Type3TestParam {
 using TupleType = std::tuple<IntType, IntType, IntType, double, double, int, bool>;
 Type3TestParam(const TupleType& t)
     : dim(std::get<0>(t)),
       num_in(std::get<1>(t)),
       num_out(std::get<2>(t)),
       upsampfac(std::get<3>(t)),
       tol(std::get<4>(t)),
       sign(std::get<5>(t)),
       use_gpu(std::get<6>(t)) {

       }

 IntType dim;
 IntType num_in;
 IntType num_out;
 double upsampfac;
 double tol;
 int sign;
 bool use_gpu;
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

template <typename T>
void compare_t3(bool use_gpu, int sign, double upsampfac, double tol, IntType num_in,
                const T* in_points_x, const T* in_points_y, const T* in_points_z, IntType num_out,
                const T* out_points_x, const T* out_points_y, const T* out_points_z,
                int batch_size = 1) {
  Options opt;
  opt.upsampfac = upsampfac;
  opt.tol = tol;

  std::minstd_rand rand_gen(42);

  auto input = rand_vec_cpx<T>(rand_gen, num_in * batch_size, -4, 4);

  std::vector<std::complex<T>> output(num_out * batch_size);
  std::vector<std::complex<T>> output_ref(num_out * batch_size);

  for (int idx_batch = 0; idx_batch < batch_size; ++idx_batch) {
    if (!in_points_y) {
      // DIM == 1
      test::nuft_direct_t3<T, 1>(sign, num_in, {in_points_x}, input.data() + idx_batch * num_in,
                                 num_out, {out_points_x}, output_ref.data() + idx_batch * num_out);

    } else if (!in_points_z) {
      // DIM == 2
      test::nuft_direct_t3<T, 2>(
          sign, num_in, {in_points_x, in_points_y}, input.data() + idx_batch * num_in, num_out,
          {out_points_x, out_points_y}, output_ref.data() + idx_batch * num_out);
    } else {
      // DIM == 3
      test::nuft_direct_t3<T, 3>(sign, num_in, {in_points_x, in_points_y, in_points_z},
                                 input.data() + idx_batch * num_in, num_out,
                                 {out_points_x, out_points_y, out_points_z},
                                 output_ref.data() + idx_batch * num_out);
    }
  }

  if(!use_gpu) {
    if (!in_points_y) {
      // DIM == 1
      PlanT3<T, 1> plan(opt, sign, num_in, {in_points_x}, num_out, {out_points_x}, batch_size);
      plan.add_input(input.data(), num_in);
      plan.transform(output.data(), num_out);

    } else if (!in_points_z) {
      // DIM == 2

      PlanT3<T, 2> plan(opt, sign, num_in, {in_points_x, in_points_y}, num_out,
                        {out_points_x, out_points_y}, batch_size);
      plan.add_input(input.data(), num_in);
      plan.transform(output.data(), num_out);

    } else {
      // DIM == 3

      PlanT3<T, 3> plan(opt, sign, num_in, {in_points_x, in_points_y, in_points_z}, num_out,
                        {out_points_x, out_points_y, out_points_z}, batch_size);
      plan.add_input(input.data(), num_in);
      plan.transform(output.data(), num_out);
    }
  } else {
#if defined(NEONUFFT_CUDA) || defined(NEONUFFT_ROCM)
    gpu::StreamType stream = nullptr;
    std::shared_ptr<Allocator> device_alloc(new gpu::DeviceAllocator());

    gpu::DeviceArray<gpu::ComplexType<T>, 1> input_device(input.size(), device_alloc);
    gpu::DeviceArray<gpu::ComplexType<T>, 1> output_device(output.size(), device_alloc);
    gpu::DeviceArray<T, 2> in_points_device({num_in, 3}, device_alloc);
    gpu::DeviceArray<T, 2> out_points_device({num_out, 3}, device_alloc);

    gpu::memcopy(ConstHostView<std::complex<T>, 1>(input.data(), input.size(), 1), input_device,
                 stream);
    gpu::memcopy(ConstHostView<T, 1>(in_points_x, num_in, 1), in_points_device.slice_view(0),
            stream);
    gpu::memcopy(ConstHostView<T, 1>(out_points_x, num_out, 1), out_points_device.slice_view(0),
            stream);

    if (!in_points_y) {
      // DIM == 1

      gpu::PlanT3<T, 1> plan(opt, sign, num_in, {in_points_device.slice_view(0).data()}, num_out,
                             {out_points_device.slice_view(0).data()}, stream, batch_size);
      plan.add_input(input_device.data(), num_in);
      plan.transform(output_device.data(), num_out);

    } else if (!in_points_z) {
      // DIM == 2
      gpu::memcopy(ConstHostView<T, 1>(in_points_y, num_in, 1),
                   in_points_device.slice_view(1), stream);
      gpu::memcopy(ConstHostView<T, 1>(out_points_y, num_out, 1),
                   out_points_device.slice_view(1), stream);

      gpu::PlanT3<T, 2> plan(
          opt, sign, num_in,
          {in_points_device.slice_view(0).data(), in_points_device.slice_view(1).data()}, num_out,
          {out_points_device.slice_view(0).data(), out_points_device.slice_view(1).data()}, stream, batch_size);
      plan.add_input(input_device.data(), num_in);
      plan.transform(output_device.data(), num_out);

    } else {
      // DIM == 3

      gpu::memcopy(ConstHostView<T, 1>(in_points_y, num_in, 1),
                   in_points_device.slice_view(1), stream);
      gpu::memcopy(ConstHostView<T, 1>(in_points_z, num_in, 1),
                   in_points_device.slice_view(2), stream);
      gpu::memcopy(ConstHostView<T, 1>(out_points_y, num_out, 1),
                   out_points_device.slice_view(1), stream);
      gpu::memcopy(ConstHostView<T, 1>(out_points_z, num_out, 1),
                   out_points_device.slice_view(2), stream);
      gpu::PlanT3<T, 3> plan(
          opt, sign, num_in,
          {in_points_device.slice_view(0).data(), in_points_device.slice_view(1).data(),
           in_points_device.slice_view(2).data()},
          num_out,
          {out_points_device.slice_view(0).data(), out_points_device.slice_view(1).data(),
           out_points_device.slice_view(2).data()},
          stream, batch_size);
      plan.add_input(input_device.data(), num_in);
      plan.transform(output_device.data(), num_out);
    }

    gpu::memcopy(output_device, HostView<std::complex<T>, 1>(output.data(), output.size(), 1),
                 stream);

    gpu::api::stream_synchronize(stream);
#else
    ASSERT_TRUE(false);
#endif
  }

  for (int idx_batch = 0; idx_batch < batch_size; ++idx_batch) {
    IntType count_below_tol = 0;
    const auto* out_batch = output.data() + idx_batch * batch_size;
    const auto* out_ref_batch = output_ref.data() + idx_batch * batch_size;
    for (IntType i = 0; i < num_out; ++i) {
      T rel_error_real =
          out_ref_batch[i].real()
              ? std::abs(out_batch[i].real() - out_ref_batch[i].real()) / out_ref_batch[i].real()
              : out_batch[i].real();
      T rel_error_imag = out_ref_batch[i].imag() ? std::abs(out_batch[i].imag() - out_ref_batch[i].imag()) /
                                                    out_ref_batch[i].imag()
                                              : out_batch[i].imag();

      const auto relative_error = std::norm(out_batch[i] - out_ref_batch[i]) / std::norm(out_ref_batch[i]);

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
}

} // namespace

template <typename T>
class Type3Test : public ::testing::TestWithParam<Type3TestParam::TupleType> {
public:
  void test_random() {
    Type3TestParam param(GetParam());
    std::minstd_rand rand_gen(42);

    auto in_points_x = rand_vec<T>(rand_gen, param.num_in, 20, 30);
    auto out_points_x = rand_vec<T>(rand_gen, param.num_out, 20, 30);
    if (param.dim == 1) {
      compare_t3<T>(param.use_gpu, param.sign, param.upsampfac, param.tol, param.num_in,
                    in_points_x.data(), nullptr, nullptr, param.num_out, out_points_x.data(),
                    nullptr, nullptr);
    } else {
      auto in_points_y = rand_vec<T>(rand_gen, param.num_in, -20, 20);
      auto out_points_y = rand_vec<T>(rand_gen, param.num_out, -20, 20);

      if (param.dim == 2) {
        compare_t3<T>(param.use_gpu, param.sign, param.upsampfac, param.tol, param.num_in,
                      in_points_x.data(), in_points_y.data(), nullptr, param.num_out,
                      out_points_x.data(), out_points_y.data(), nullptr);
      } else {
        auto in_points_z = rand_vec<T>(rand_gen, param.num_in, -20, -15);
        auto out_points_z = rand_vec<T>(rand_gen, param.num_out, -20, -15);
        compare_t3<T>(param.use_gpu, param.sign, param.upsampfac, param.tol, param.num_in,
                      in_points_x.data(), in_points_y.data(), in_points_z.data(), param.num_out,
                      out_points_x.data(), out_points_y.data(), out_points_z.data());
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
  compare_t3<double>(false, -1, 2, 1e-15, points_x.size(), points_x.data(), nullptr, nullptr,
                     points_x.size(), points_x.data(), nullptr, nullptr);
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
  compare_t3<double>(false, -1, 2, 1e-15, points_x.size(), points_x.data(), points_y.data(),
                     nullptr, points_x.size(), points_x.data(), points_y.data(), nullptr);
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
  compare_t3<double>(false, -1, 2, 1e-15, points_x.size(), points_x.data(), points_y.data(),
                     points_z.data(), points_x.size(), points_x.data(), points_y.data(),
                     points_z.data());
}

TEST(Type3TestBatched, d1) {
  const int batch_size = 3;
  const IntType num_in = 200;
  const IntType num_out = 55;
  std::minstd_rand rand_gen(42);
  auto in_points_x = rand_vec<double>(rand_gen, num_in, 20, 30);
  auto out_points_x = rand_vec<double>(rand_gen, num_out, 20, 30);
  compare_t3<double>(false, -1, 2, 1e-6, in_points_x.size(), in_points_x.data(), nullptr, nullptr,
                     out_points_x.size(), out_points_x.data(), nullptr, nullptr, batch_size);
}

TEST(Type3TestBatched, d2) {
  const int batch_size = 3;
  const IntType num_in = 80;
  const IntType num_out = 55;
  std::minstd_rand rand_gen(42);
  auto in_points_x = rand_vec<double>(rand_gen, num_in, 20, 30);
  auto out_points_x = rand_vec<double>(rand_gen, num_out, 20, 30);
  auto in_points_y = rand_vec<double>(rand_gen, num_in, -20, 20);
  auto out_points_y = rand_vec<double>(rand_gen, num_out, -20, 20);
  compare_t3<double>(false, -1, 2, 1e-6, in_points_x.size(), in_points_x.data(),
                     in_points_y.data(), nullptr, out_points_x.size(), out_points_x.data(),
                     out_points_y.data(), nullptr, batch_size);
}

//TODO: fix nan on some systems
// TEST(Type3TestBatched, d3) {
//   const int batch_size = 3;
//   const IntType num_in = 80;
//   const IntType num_out = 55;
//   std::minstd_rand rand_gen(42);
//   auto in_points_x = rand_vec<double>(rand_gen, num_in, 20, 30);
//   auto out_points_x = rand_vec<double>(rand_gen, num_out, 20, 30);
//   auto in_points_y = rand_vec<double>(rand_gen, num_in, -20, 20);
//   auto out_points_y = rand_vec<double>(rand_gen, num_out, -20, 20);
//   auto in_points_z = rand_vec<double>(rand_gen, num_in, -20, -15);
//   auto out_points_z = rand_vec<double>(rand_gen, num_out, -20, -15);
//   compare_t3<double>(false, -1, 2, 1e-6, in_points_x.size(), in_points_x.data(),
//                      in_points_y.data(), out_points_z.data(), out_points_x.size(),
//                      out_points_x.data(), out_points_y.data(), out_points_z.data(), batch_size);
// }


static auto param_type_names(
    const ::testing::TestParamInfo<Type3TestParam::TupleType> &info)
    -> std::string {
  std::stringstream stream;

  Type3TestParam param(info.param);

  if (param.use_gpu)
    stream << "gpu_";
  else
    stream << "host_";
  stream << "d_" << param.dim;
  stream << "_m_" << param.num_in;
  stream << "_n_" << param.num_out;
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
    Type3, Type3TestFloat,
    ::testing::Combine(::testing::Values<IntType>(1, 2),             // dimension
                       ::testing::Values<IntType>(1, 10, 200, 503),  // number of in points
                       ::testing::Values<IntType>(1, 10, 200, 503),  // number of out points
                       ::testing::Values<double>(2.0),               // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-7),        // tolerance
                       ::testing::Values<int>(1, -1),                // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type3, Type3TestDouble,
    ::testing::Combine(::testing::Values<IntType>(1, 2),             // dimension
                       ::testing::Values<IntType>(1, 10, 200, 503),  // number of in points
                       ::testing::Values<IntType>(1, 10, 200, 503),  // number of out points
                       ::testing::Values<double>(2.0),               // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-7),        // tolerance
                       ::testing::Values<int>(1, -1),                // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type3D3, Type3TestFloat,
    ::testing::Combine(::testing::Values<IntType>(3),          // dimension
                       ::testing::Values<IntType>(1, 10, 99),  // number of in points
                       ::testing::Values<IntType>(1, 10, 99),  // number of out points
                       ::testing::Values<double>(2.0),         // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-7),  // tolerance
                       ::testing::Values<int>(1),              // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Type3D3, Type3TestDouble,
    ::testing::Combine(::testing::Values<IntType>(3),          // dimension
                       ::testing::Values<IntType>(1, 10, 99),  // number of in points
                       ::testing::Values<IntType>(1, 10, 99),  // number of out points
                       ::testing::Values<double>(2.0),         // upsampling factor
                       ::testing::Values<double>(1e-4, 1e-7),  // tolerance
                       ::testing::Values<int>(1),              // sign
                       ::testing::Values<bool>(NEONUFFT_PU_VALUES)),
    param_type_names);
