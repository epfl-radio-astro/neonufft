# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/neonufftSharedConfigVersion.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftSharedConfigVersion.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftStaticConfigVersion.cmake")
endif()
