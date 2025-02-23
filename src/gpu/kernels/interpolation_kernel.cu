#include "neonufft//config.h"
//---
#include <algorithm>

#include "es_kernel_param.hpp"
#include "gpu/kernels/interpolation_kernel.hpp"
#include "gpu/memory/device_view.hpp"
#include "gpu/util/cub_api.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"
#include "neonufft/gpu/types.hpp"
#include "util/point.hpp"

namespace neonufft {
namespace gpu {

template <typename T, int N_SPREAD>
struct EsKernelDirect;

template <int N_SPREAD>
struct EsKernelDirect<float, N_SPREAD> {
  __device__ __forceinline__ float eval_scalar(float x) const {
    constexpr float es_c = 4.f / float(N_SPREAD * N_SPREAD);
    const float arg = 1.f - es_c * x * x;
    if (arg <= 0) return 0.0;

    return expf(es_beta * (sqrtf(arg) - 1.f));
  }

  float es_beta;
};

template <int N_SPREAD>
struct EsKernelDirect<double, N_SPREAD> {
  __device__ __forceinline__ double eval_scalar(double x) const {
    constexpr float es_c = 4.0 / double(N_SPREAD * N_SPREAD);
    const double arg = 1.0 - es_c * x * x;
    if (arg <= 0) return 0.0;

    return exp(es_beta * (sqrt(arg) - 1.0));
  }

  double es_beta;
};

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    interpolation_1d_kernel(KER kernel, ConstDeviceView<Point<T, 1>, 1> points,
                            ConstDeviceView<ComplexType<T>, 1> grid,
                            DeviceView<ComplexType<T>, 1> out) {
  const IntType block_step = gridDim.x * BLOCK_SIZE;
  const T half_width = T(N_SPREAD) / T(2);  // half spread width

  for (IntType idx = threadIdx.x + blockIdx.x * BLOCK_SIZE; idx < points.shape(0);
       idx += block_step) {
    const auto point = points[idx];
    const T loc = point.coord[0] * grid.shape(0);  // px in [0, 1]
    const T idx_init_ceil = ceil(loc - half_width);
    IntType idx_init = IntType(idx_init_ceil);  // fine padded_grid start index (-
    T x = idx_init_ceil - loc;                  // x1 in [-w/2,-w/2+1], up to rounding

    int idx_ker = 0;
    IntType idx_grid = idx_init;

    ComplexType<T> sum{0, 0};

    // wrap left
    for (; idx_ker < N_SPREAD && idx_grid < 0; ++idx_ker, ++idx_grid) {
      const auto ker_val = kernel.eval_scalar(x + idx_ker);
      const auto grid_val = grid[grid.size() + idx_grid];

      sum.x += ker_val * grid_val.x;
      sum.y += ker_val * grid_val.y;
    }

    // inner
    for (; idx_ker < N_SPREAD && idx_grid < grid.size(); ++idx_ker, ++idx_grid) {
      const auto ker_val = kernel.eval_scalar(x + idx_ker);
      const auto grid_val = grid[idx_grid];

      sum.x += ker_val * grid_val.x;
      sum.y += ker_val * grid_val.y;
    }

    // wrap right
    for (; idx_ker < N_SPREAD; ++idx_ker, ++idx_grid) {
      const auto ker_val = kernel.eval_scalar(x + idx_ker);
      const auto grid_val = grid[idx_grid - grid.size()];

      sum.x += ker_val * grid_val.x;
      sum.y += ker_val * grid_val.y;
    }

    out[point.index] = sum;
  }
}

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE, int WARP_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    interpolation_2d_kernel(KER kernel, ConstDeviceView<Point<T, 2>, 1> points,
                            ConstDeviceView<ComplexType<T>, 2> grid,
                            DeviceView<ComplexType<T>, 1> out) {
  static_assert(2 * N_SPREAD <= WARP_SIZE);

  constexpr int n_warps_per_block = BLOCK_SIZE / WARP_SIZE;

  using WarpReduce = typename cub_api::WarpReduce<T>;
  __shared__ typename WarpReduce::TempStorage temp_storage[n_warps_per_block];

  __shared__ T ker_storage[2 * N_SPREAD * n_warps_per_block];

  const int n_warps_global = n_warps_per_block * gridDim.x;
  const int local_warp_id = threadIdx.x / WARP_SIZE;
  const int global_warp_id = local_warp_id + blockIdx.x * n_warps_per_block;
  const int warp_thread_id = threadIdx.x % WARP_SIZE;

  constexpr int ker_step_size_y = WARP_SIZE / N_SPREAD;
  const int col_init = warp_thread_id / N_SPREAD;
  const int idx_ker_x = warp_thread_id % N_SPREAD;

  const IntType points_per_warp = (points.shape(0) + n_warps_global - 1) / n_warps_global;
  const IntType idx_point_init = global_warp_id * points_per_warp;

  T* ker = ker_storage + 2 * N_SPREAD * local_warp_id;
  T* ker_x = ker;
  T* ker_y = ker + N_SPREAD;

  const T half_width = T(N_SPREAD) / T(2);  // half spread width

  // iterate over points, such that close points are processed by close warps. Helps with caching if
  // points are spatially sorted.
  for (IntType idx = idx_point_init;
       idx < points.shape(0) && idx < idx_point_init + points_per_warp; ++idx) {
    ComplexType<T> sum{0, 0};

    const auto point = points[idx];
    const T loc_x = point.coord[0] * grid.shape(0);  // px in [0, 1]
    const T idx_init_ceil_x = ceil(loc_x - half_width);
    const T loc_y = point.coord[1] * grid.shape(1);  // px in [0, 1]
    const T idx_init_ceil_y = ceil(loc_y - half_width);

    IntType idx_init_x = IntType(idx_init_ceil_x);  // fine padded_grid start index (-
    T p_x = idx_init_ceil_x - loc_x;                // x1 in [-w/2,-w/2+1], up to rounding
                                                    //
    IntType idx_init_y = IntType(idx_init_ceil_y);  // fine padded_grid start index (-
    T p_y = idx_init_ceil_y - loc_y;                // x1 in [-w/2,-w/2+1], up to rounding

    // precompute kernel values. [0, N_SPREAD) threads compute along x and [N_SPREAD, 2*N_SPREAD)
    // along y axis
    if (warp_thread_id < 2 * N_SPREAD) {
      T ker_input =
          warp_thread_id < N_SPREAD ? p_x + warp_thread_id : p_y + (warp_thread_id - N_SPREAD);

      // expensive kernel evaluation
      ker[warp_thread_id] = kernel.eval_scalar(ker_input);
    }
    // sync not available / required on AMD
#if defined(__CUDACC__)
    __syncwarp();
#endif

    if (warp_thread_id < ker_step_size_y * N_SPREAD && col_init < N_SPREAD) {
      const T ker_value_x = ker_x[idx_ker_x];
      auto grid_idx_x = idx_init_x + idx_ker_x;
      if (grid_idx_x < 0)
        grid_idx_x += grid.shape(0);
      else if (grid_idx_x >= grid.shape(0))
        grid_idx_x -= grid.shape(0);

      for (int idx_ker_y = col_init; idx_ker_y < N_SPREAD; idx_ker_y += ker_step_size_y) {
        auto grid_idx_y = idx_init_y + idx_ker_y;
        if (grid_idx_y < 0)
          grid_idx_y += grid.shape(1);
        else if (grid_idx_y >= grid.shape(1))
          grid_idx_y -= grid.shape(1);

        const auto grid_value = grid[{grid_idx_x, grid_idx_y}];

        const T ker_value = ker_value_x * ker_y[idx_ker_y];
        sum.x += ker_value * grid_value.x;
        sum.y += ker_value * grid_value.y;
      }
    }

    T sum_real = WarpReduce(temp_storage[local_warp_id]).Sum(sum.x);
#if defined(__CUDACC__)
    __syncwarp();
#endif

    T sum_imag = WarpReduce(temp_storage[local_warp_id]).Sum(sum.y);
#if defined(__CUDACC__)
    __syncwarp();
#endif

    if (warp_thread_id == 0) {
      out[point.index] = ComplexType<T>{sum_real, sum_imag};
    }
  }
}

template <typename KER, typename T, int N_SPREAD, int BLOCK_SIZE, int WARP_SIZE>
__global__ static void __launch_bounds__(BLOCK_SIZE)
    interpolation_3d_kernel(KER kernel, ConstDeviceView<Point<T, 3>, 1> points,
                            ConstDeviceView<ComplexType<T>, 3> grid,
                            DeviceView<ComplexType<T>, 1> out) {
  static_assert(2 * N_SPREAD <= WARP_SIZE);

  constexpr int n_warps_per_block = BLOCK_SIZE / WARP_SIZE;

  using WarpReduce = typename cub_api::WarpReduce<T>;
  __shared__ typename WarpReduce::TempStorage temp_storage[n_warps_per_block];

  __shared__ T ker_storage[3 * N_SPREAD * n_warps_per_block];

  const int n_warps_global = n_warps_per_block * gridDim.x;
  const int local_warp_id = threadIdx.x / WARP_SIZE;
  const int global_warp_id = local_warp_id + blockIdx.x * n_warps_per_block;
  const int warp_thread_id = threadIdx.x % WARP_SIZE;

  constexpr int ker_step_size_y = WARP_SIZE / N_SPREAD;
  const int col_init = warp_thread_id / N_SPREAD;
  const int idx_ker_x = warp_thread_id % N_SPREAD;

  const IntType points_per_warp = (points.shape(0) + n_warps_global - 1) / n_warps_global;
  const IntType idx_point_init = global_warp_id * points_per_warp;

  T* ker = ker_storage + 3 * N_SPREAD * local_warp_id;
  T* ker_x = ker;
  T* ker_y = ker + N_SPREAD;
  T* ker_z = ker + 2 * N_SPREAD;

  const T half_width = T(N_SPREAD) / T(2);  // half spread width

  // iterate over points, such that close points are processed by close warps. Helps with caching if
  // points are spatially sorted.
  for (IntType idx = idx_point_init;
       idx < points.shape(0) && idx < idx_point_init + points_per_warp; ++idx) {
    ComplexType<T> sum{0, 0};

    const auto point = points[idx];
    const T loc_x = point.coord[0] * grid.shape(0);  // px in [0, 1]
    const T idx_init_ceil_x = ceil(loc_x - half_width);
    IntType idx_init_x = IntType(idx_init_ceil_x);  // fine padded_grid start index (-
    T p_x = idx_init_ceil_x - loc_x;                // x1 in [-w/2,-w/2+1], up to rounding

    const T loc_y = point.coord[1] * grid.shape(1);  // px in [0, 1]
    const T idx_init_ceil_y = ceil(loc_y - half_width);
    IntType idx_init_y = IntType(idx_init_ceil_y);   // fine padded_grid start index (-
    T p_y = idx_init_ceil_y - loc_y;                 // x1 in [-w/2,-w/2+1], up to rounding
                                                     //
    const T loc_z = point.coord[2] * grid.shape(2);  // px in [0, 1]
    const T idx_init_ceil_z = ceil(loc_z - half_width);
    IntType idx_init_z = IntType(idx_init_ceil_z);  // fine padded_grid start index (-
    T p_z = idx_init_ceil_z - loc_z;                // x1 in [-w/2,-w/2+1], up to rounding

    // precompute kernel values. [0, N_SPREAD) threads compute along x and [N_SPREAD, 2*N_SPREAD)
    // along y axis
    if constexpr (WARP_SIZE <= 3 * N_SPREAD) {
      if (warp_thread_id < 3 * N_SPREAD) {
        // select input for thread partitioned into [0, N_SPREAD), [N_SPREAD, 2*N_SPREAD) and
        // [N_SPREAD, 3*N_SPREAD)
        T ker_input = warp_thread_id < N_SPREAD
                          ? p_x + warp_thread_id
                          : (warp_thread_id < 2 * N_SPREAD ? p_y + (warp_thread_id - N_SPREAD)
                                                           : p_z + (warp_thread_id - 2 * N_SPREAD));

        // expensive kernel evaluation
        ker[warp_thread_id] = kernel.eval_scalar(ker_input);
      }
    } else {
      if (warp_thread_id < 2 * N_SPREAD) {
        T ker_input =
            warp_thread_id < N_SPREAD ? p_x + warp_thread_id : p_y + (warp_thread_id - N_SPREAD);

        // expensive kernel evaluation
        ker[warp_thread_id] = kernel.eval_scalar(ker_input);
      }
      if (warp_thread_id < N_SPREAD) {
        T ker_input = p_z + warp_thread_id;

        // expensive kernel evaluation
        ker[warp_thread_id + 2 * N_SPREAD] = kernel.eval_scalar(ker_input);
      }
    }
    // sync not available / required on AMD
#if defined(__CUDACC__)
    __syncwarp();
#endif

    if (warp_thread_id < ker_step_size_y * N_SPREAD && col_init < N_SPREAD) {
      const T ker_value_x = ker_x[idx_ker_x];
      auto grid_idx_x = idx_init_x + idx_ker_x;
      if (grid_idx_x < 0)
        grid_idx_x += grid.shape(0);
      else if (grid_idx_x >= grid.shape(0))
        grid_idx_x -= grid.shape(0);

      // TODO: optimize for non-wrapped case
      for (int idx_ker_z = 0; idx_ker_z < N_SPREAD; ++idx_ker_z) {
        const T ker_value_xz = ker_value_x * ker_z[idx_ker_z];
        auto grid_idx_z = idx_init_z + idx_ker_z;
        if (grid_idx_z < 0)
          grid_idx_z += grid.shape(2);
        else if (grid_idx_z >= grid.shape(2))
          grid_idx_z -= grid.shape(2);

        for (int idx_ker_y = col_init; idx_ker_y < N_SPREAD; idx_ker_y += ker_step_size_y) {
          auto grid_idx_y = idx_init_y + idx_ker_y;
          if (grid_idx_y < 0)
            grid_idx_y += grid.shape(1);
          else if (grid_idx_y >= grid.shape(1))
            grid_idx_y -= grid.shape(1);

          const auto grid_value = grid[{grid_idx_x, grid_idx_y, grid_idx_z}];

          const T ker_value = ker_value_xz * ker_y[idx_ker_y];
          sum.x += ker_value * grid_value.x;
          sum.y += ker_value * grid_value.y;
        }
      }
    }

    T sum_real = WarpReduce(temp_storage[local_warp_id]).Sum(sum.x);
#if defined(__CUDACC__)
    __syncwarp();
#endif

    T sum_imag = WarpReduce(temp_storage[local_warp_id]).Sum(sum.y);
#if defined(__CUDACC__)
    __syncwarp();
#endif

    if (warp_thread_id == 0) {
      out[point.index] = ComplexType<T>{sum_real, sum_imag};
    }
  }
}

template <typename T, IntType DIM, int N_SPREAD, int BLOCK_SIZE, int WARP_SIZE>
auto interpolation_dispatch(const api::DevicePropType& prop, const api::StreamType& stream,
                            const KernelParameters<T>& param,
                            ConstDeviceView<Point<T, DIM>, 1> points,
                            ConstDeviceView<ComplexType<T>, DIM> grid,
                            DeviceView<ComplexType<T>, 1> out) -> void {
  static_assert(N_SPREAD >= 2);
  static_assert(N_SPREAD <= 16);

  if (param.n_spread == N_SPREAD) {
    EsKernelDirect<T, N_SPREAD> kernel{param.es_beta};
    const dim3 block_dim(std::min<int>(BLOCK_SIZE, prop.maxThreadsDim[0]), 1, 1);
    const auto grid_dim = kernel_launch_grid(prop, {points.size(), 1, 1}, block_dim);

    if constexpr (DIM == 1) {
      api::launch_kernel(interpolation_1d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE>,
                         grid_dim, block_dim, 0, stream, kernel, points, grid, out);
    } else if constexpr (DIM == 2) {
      api::launch_kernel(
          interpolation_2d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE, WARP_SIZE>, grid_dim,
          block_dim, 0, stream, kernel, points, grid, out);
    } else {
      api::launch_kernel(
          interpolation_3d_kernel<decltype(kernel), T, N_SPREAD, BLOCK_SIZE, WARP_SIZE>, grid_dim,
          block_dim, 0, stream, kernel, points, grid, out);
    }
  } else {
    if constexpr (N_SPREAD > 2) {
      interpolation_dispatch<T, DIM, N_SPREAD - 1, BLOCK_SIZE, WARP_SIZE>(prop, stream, param,
                                                                          points, grid, out);
    } else {
      throw InternalError("n_spread not in [2, 16]");
    }
  }
}

template <typename T, IntType DIM>
auto interpolation(const api::DevicePropType& prop, const api::StreamType& stream,
                   const KernelParameters<T>& param, ConstDeviceView<Point<T, DIM>, 1> points,
                   ConstDeviceView<ComplexType<T>, DIM> grid, DeviceView<ComplexType<T>, 1> out)
    -> void {
  constexpr int block_size = 128;
  if (prop.warpSize == 32) {
    interpolation_dispatch<T, DIM, 16, block_size, 32>(prop, stream, param, points, grid, out);
  } else if (prop.warpSize == 64) {
    interpolation_dispatch<T, DIM, 16, block_size, 64>(prop, stream, param, points, grid, out);
  } else {
    throw GPUError("Unsupported GPU warp size.");
  }
}

template auto interpolation<float, 1>(const api::DevicePropType& prop,
                                      const api::StreamType& stream,
                                      const KernelParameters<float>& param,
                                      ConstDeviceView<Point<float, 1>, 1> points,
                                      ConstDeviceView<ComplexType<float>, 1> grid,
                                      DeviceView<ComplexType<float>, 1> out) -> void;

template auto interpolation<float, 2>(const api::DevicePropType& prop,
                                      const api::StreamType& stream,
                                      const KernelParameters<float>& param,
                                      ConstDeviceView<Point<float, 2>, 1> points,
                                      ConstDeviceView<ComplexType<float>, 2> grid,
                                      DeviceView<ComplexType<float>, 1> out) -> void;

template auto interpolation<float, 3>(const api::DevicePropType& prop,
                                      const api::StreamType& stream,
                                      const KernelParameters<float>& param,
                                      ConstDeviceView<Point<float, 3>, 1> points,
                                      ConstDeviceView<ComplexType<float>, 3> grid,
                                      DeviceView<ComplexType<float>, 1> out) -> void;

template auto interpolation<double, 1>(const api::DevicePropType& prop,
                                       const api::StreamType& stream,
                                       const KernelParameters<double>& param,
                                       ConstDeviceView<Point<double, 1>, 1> points,
                                       ConstDeviceView<ComplexType<double>, 1> grid,
                                       DeviceView<ComplexType<double>, 1> out) -> void;

template auto interpolation<double, 2>(const api::DevicePropType& prop,
                                       const api::StreamType& stream,
                                       const KernelParameters<double>& param,
                                       ConstDeviceView<Point<double, 2>, 1> points,
                                       ConstDeviceView<ComplexType<double>, 2> grid,
                                       DeviceView<ComplexType<double>, 1> out) -> void;

template auto interpolation<double, 3>(const api::DevicePropType& prop,
                                       const api::StreamType& stream,
                                       const KernelParameters<double>& param,
                                       ConstDeviceView<Point<double, 3>, 1> points,
                                       ConstDeviceView<ComplexType<double>, 3> grid,
                                       DeviceView<ComplexType<double>, 1> out) -> void;

}  // namespace gpu
}  // namespace neonufft
