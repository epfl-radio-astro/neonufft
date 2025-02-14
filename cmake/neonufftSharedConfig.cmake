include(CMakeFindDependencyMacro)
macro(find_dependency_components)
	if(${ARGV0}_FOUND AND ${CMAKE_VERSION} VERSION_LESS "3.15.0")
		# find_dependency does not handle new components correctly before 3.15.0
		set(${ARGV0}_FOUND FALSE)
	endif()
	find_dependency(${ARGV})
endmacro()

# find_dependency may set neonufft_FOUND to false, so only add neonufft if everything required was found
if(NOT DEFINED neonufft_FOUND OR neonufft_FOUND)
	# add version of package
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftSharedConfigVersion.cmake")

	# add library target
	include("${CMAKE_CURRENT_LIST_DIR}/neonufftSharedTargets.cmake")
endif()
