include(CMakeFindDependencyMacro)

if(NEONUFFT_CUDA)
	find_dependency(CUDAToolkit MODULE)
endif()

if(NEONUFFT_ROCM)
  message(FATAL_ERROR "ROCM backend not yet implemented")
endif()

# find_dependency may set neonufft_FOUND to false, so only add neonufft if everything required was found
if(NOT DEFINED neonufft_FOUND OR neonufft_FOUND)
	# add version of package
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftSharedConfigVersion.cmake")

	# add library target
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftSharedTargets.cmake")
endif()
