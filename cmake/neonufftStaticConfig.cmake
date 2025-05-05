include(CMakeFindDependencyMacro)

find_dependency(FFTW MODULE)
find_dependency(FFTWF MODULE)
find_dependency(hwy CONFIG)
if(NEONUFFT_OMP)
  find_dependency(OpenMP MODULE)
elseif(NEONUFFT_TBB)
  find_dependency(TBB CONFIG)
endif()

if(NEONUFFT_CUDA)
	find_dependency(CUDAToolkit MODULE)
endif()

if(NEONUFFT_ROCM)
  find_dependency(hip CONFIG)
	find_dependency(hipfft CONFIG)
  find_dependency(hipcub CONFIG)
endif()

# find_dependency may set neonufft_FOUND to false, so only add neonufft if everything required was found
if(NOT DEFINED neonufft_FOUND OR neonufft_FOUND)
	# add version of package
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftStaticConfigVersion.cmake")

	# add library target
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftStaticTargets.cmake")
endif()

