cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
file(READ "VERSION" NEONUFFT_VERSION)
string(STRIP ${NEONUFFT_VERSION} NEONUFFT_VERSION)
project(neonufft LANGUAGES CXX VERSION "${NEONUFFT_VERSION}")
set(NEONUFFT_SO_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})

# allow {module}_ROOT variables to be set
if(POLICY CMP0074)
	cmake_policy(SET CMP0074 NEW)
endif()

# use INTERFACE_LINK_LIBRARIES property if available
if(POLICY CMP0022)
	cmake_policy(SET CMP0022 NEW)
endif()

# update time stamps when using FetchContent
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

include(CMakeDependentOption)

if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(_DEFAULT_THREADING "NATIVE")
else()
  set(_DEFAULT_THREADING "OPENMP")
endif()

# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
	)
endif()

set(NEONUFFT_BUILD_TYPE "OFF" CACHE STRING "If set, overrides the CMAKE_BUILD_TYPE variable.")
set_property(CACHE NEONUFFT_BUILD_TYPE PROPERTY STRINGS
	"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
if(NEONUFFT_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE ${NEONUFFT_BUILD_TYPE} CACHE STRING "Build type" FORCE)
endif()

option(NEONUFFT_MULTI_ARCH "Build kernels for multiple CPU architectues with dynamic dispatch. When disabled, arch flags should be set through CMAKE_CXX_FLAGS." ON)

option(NEONUFFT_BUNDLED_LIBS "Use all bundled libraries" ON)
cmake_dependent_option(NEONUFFT_BUNDLED_HIGHWAY "Use bundled highway lib" ON "NEONUFFT_BUNDLED_LIBS" OFF)
cmake_dependent_option(NEONUFFT_BUNDLED_GOOGLETEST "Use bundled google test lib" ON "NEONUFFT_BUNDLED_LIBS" OFF)
option(NEONUFFT_BUILD_TESTS "Build tests" OFF)
option(NEONUFFT_INSTALL_LIB "Install library" ON)

set(NEONUFFT_GPU "OFF" CACHE STRING "GPU backend")
set_property(CACHE NEONUFFT_GPU PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)

set(NEONUFFT_THREADING ${_DEFAULT_THREADING} CACHE STRING "Threading backend.")
set_property(CACHE NEONUFFT_THREADING PROPERTY STRINGS "OPENMP" "TBB" "NATIVE")

set(NEONUFFT_FLAGS "")
set(NEONUFFT_EXTERNAL_LIBS "")
set(NEONUFFT_EXTERNAL_LIBS_PUBLIC "")
set(NEONUFFT_INCLUDE_DIRS "")
set(NEONUFFT_GPU_EXTERNAL_LIBS "")
set(NEONUFFT_GPU_EXTERNAL_LIBS_PUBLIC "")

set(NEONUFFT_CUDA OFF)
set(NEONUFFT_ROCM OFF)
if(NEONUFFT_GPU)
	if(NEONUFFT_GPU STREQUAL "CUDA")
		set(NEONUFFT_CUDA ON)
	elseif(NEONUFFT_GPU STREQUAL "ROCM")
		set(NEONUFFT_ROCM ON)
		set(NEONUFFT_MAGMA ON)
	else()
		message(FATAL_ERROR "Invalid GPU backend")
	endif()
endif()

# set language and standard
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

# Get GNU standard install prefixes
include(GNUInstallDirs)

include(FetchContent)
include(CMakeDependentOption)

#add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

# Options
set(BUILD_SHARED_LIBS "ON" CACHE STRING "Build as shared library") # default to shared

find_package(FFTW MODULE REQUIRED)
list(APPEND NEONUFFT_EXTERNAL_LIBS FFTW::FFTW)
find_package(FFTWF MODULE REQUIRED)
list(APPEND NEONUFFT_EXTERNAL_LIBS FFTWF::FFTWF)

if(NEONUFFT_BUNDLED_HIGHWAY)
  # add google highway
  set(HWY_ENABLE_CONTRIB ON CACHE BOOL "")
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
  set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
  set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "")
  set(HWY_ENABLE_CONTRIB ON CACHE BOOL "")
  FetchContent_Declare(
    hwy
    URL https://github.com/google/highway/archive/refs/tags/1.2.0.tar.gz
    URL_MD5 8b3d090a2d081730b40bca5ae0d65f11
  )
  FetchContent_MakeAvailable(hwy)
  list(APPEND NEONUFFT_EXTERNAL_LIBS hwy)
else()
  find_package(hwy CONFIG REQUIRED)
  list(APPEND NEONUFFT_EXTERNAL_LIBS hwy::hwy)
endif()


set(NEONUFFT_OMP OFF)
set(NEONUFFT_TBB OFF)
string(TOUPPER ${NEONUFFT_THREADING} NEONUFFT_THREADING_UPPER)
if(NEONUFFT_THREADING_UPPER STREQUAL "OPENMP")
  set(NEONUFFT_OMP ON)
  find_package(OpenMP MODULE REQUIRED)
  list(APPEND NEONUFFT_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
elseif(NEONUFFT_THREADING_UPPER STREQUAL "TBB")
  set(NEONUFFT_TBB ON)
  find_package(TBB CONFIG REQUIRED)
  list(APPEND NEONUFFT_EXTERNAL_LIBS TBB::tbb)
elseif(NOT NEONUFFT_THREADING_UPPER STREQUAL "NATIVE")
  message(FATAL_ERROR "Invalid NEONUFFT_THREADING value")
endif()

if(NEONUFFT_CUDA)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)

	list(APPEND NEONUFFT_GPU_EXTERNAL_LIBS CUDA::cufft)
	list(APPEND NEONUFFT_GPU_EXTERNAL_LIBS_PUBLIC CUDA::cudart)
endif()

if(NEONUFFT_ROCM)
  # find_package(hipcub CONFIG REQUIRED)
  message(FATAL_ERROR "ROCM backend not yet implemented")
endif()

# generate config.h
configure_file(cmake/config.h.in ${PROJECT_BINARY_DIR}/include/neonufft/config.h)

list(APPEND NEONUFFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/contrib)
list(APPEND NEONUFFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)
list(APPEND NEONUFFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
list(APPEND NEONUFFT_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/include)

add_subdirectory(contrib)
add_subdirectory(src)
if(NEONUFFT_BUILD_TESTS)
  add_subdirectory(tests)
endif()
