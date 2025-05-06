# NeoNUFFT

NeoNUFFT is a rewrite of the [fiNUFFT](https://github.com/flatironinstitute/finufft) library with extended functionality using [highway](https://github.com/google/highway) for portable vectorization. It supports GPU usage through CUDA and ROCm.

## Installation

### CMake options
Bipp can be configured with the following options:
| Option                          |  Values                    | Default     | Description                                                                                               |
|---------------------------------|----------------------------|-------------|-----------------------------------------------------------------------------------------------------------|
| `NEONUFFT_MUTLI_ARCH`           |  `ON`, `OFF`               | `ON`        | Build for multiple CPU architecture for performance portability                                           |
| `NEONUFFT_THREADING`            |  `NATIVE`, `OPENMP`, `TBB` | `OPENMP`    | Select GPU backend                                                                                        |
| `NEONUFFT_GPU`                  |  `OFF`, `CUDA`, `ROCM`     | `OFF`       | Select GPU backend                                                                                        |
| `NEONUFFT_BUILD_TESTS`          |  `ON`, `OFF`               | `OFF`       | Build test executables                                                                                    |
| `NEONUFFT_INSTALL_LIB`          |  `ON`, `OFF`               | `ON`        | Add library to install target                                                                             |
| `NEONUFFT_BUNDLED_LIBS`         |  `ON`, `OFF`               | `ON`        | Download and build Google Highway and GoogleTest                                                          |


### Examples

Portable CPU only build:
```
cmake -B build -DNEONUFFT_THREADING=NATIVE -DNEONUFFT_GPU=OFF -DNEONUFFT_MULTI_ARCH=ON
make -C build
```


Native build with CUDA for sm_90 architecture:
```
cmake -B build -DNEONUFFT_THREADING=OPENMP -DNEONUFFT_GPU=CUDA -DNEONUFFT_MULTI_ARCH=OFF -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_CUDA_ARCHITECTURES=90
make -C build
```
