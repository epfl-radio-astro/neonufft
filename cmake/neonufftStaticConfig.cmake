include(CMakeFindDependencyMacro)

find_dependency(FFTW MODULE REQUIRED)
find_dependency(FFTWF MODULE REQUIRED)
find_dependency(hwy CONFIG REQUIRED)
if(NEONUFFT_OMP)
  find_dependency(OpenMP MODULE REQUIRED)
elseif(NEONUFFT_TBB)
  find_dependency(TBB CONFIG REQUIRED)
endif()

# find_dependency may set neonufft_FOUND to false, so only add neonufft if everything required was found
if(NOT DEFINED neonufft_FOUND OR neonufft_FOUND)
	# add version of package
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftStaticConfigVersion.cmake")

	# add library target
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftStaticTargets.cmake")
endif()

