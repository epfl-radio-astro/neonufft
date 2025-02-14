# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/neonufftSharedTargets.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftSharedTargets.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftStaticTargets.cmake")
endif()
