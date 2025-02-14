set(NEONUFFT_MPI @NEONUFFT_MPI@)
set(NEONUFFT_OMP @NEONUFFT_OMP@)
set(NEONUFFT_TBB @NEONUFFT_TBB@)

# Only look for modules we installed and save value
set(_CMAKE_MODULE_PATH_SAVE ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules")

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/neonufftSharedConfig.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftSharedConfig.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftStaticConfig.cmake")
endif()

set(CMAKE_MODULE_PATH ${_CMAKE_MODULE_PATH_SAVE}) # restore module path
