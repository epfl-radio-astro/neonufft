set(NEONUFFT_SOURCE_FILES
	plan.cpp
	kernels/interpolation_kernel.cpp
	kernels/upsample_kernel.cpp
	kernels/downsample_kernel.cpp
	kernels/rescale_loc_kernel.cpp
	kernels/spreading_kernel.cpp
	kernels/min_max_kernel.cpp
	kernels/compute_postphase_kernel.cpp
	kernels/compute_prephase_kernel.cpp
	kernels/fold_padding.cpp
	kernels/nuft_real_kernel.cpp
	threading/thread_pool.cpp
	memory/array.cpp
	util/fft_grid.cpp
	)


add_library(neonufft_objects OBJECT ${NEONUFFT_SOURCE_FILES})

# make static libraries linkable into shared ones
set_property(TARGET neonufft_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(neonufft_objects PRIVATE ${NEONUFFT_FLAGS})
target_include_directories(neonufft_objects PRIVATE ${NEONUFFT_INCLUDE_DIRS})
target_link_libraries(neonufft_objects PRIVATE ${NEONUFFT_EXTERNAL_LIBS})
target_link_libraries(neonufft_objects PRIVATE ${NEONUFFT_EXTERNAL_LIBS_PUBLIC})
target_include_directories(neonufft_objects INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
set_target_properties(neonufft_objects PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)
target_compile_options(neonufft_objects PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>) # no ABI warnings about C++17 compatability


#shared library
add_library(neonufft $<TARGET_OBJECTS:neonufft_objects> $<TARGET_OBJECTS:neonufft_contrib_objects>)

set_property(TARGET neonufft PROPERTY VERSION ${NEONUFFT_VERSION})
set_property(TARGET neonufft PROPERTY SOVERSION ${NEONUFFT_SO_VERSION})

target_link_libraries(neonufft PRIVATE ${NEONUFFT_EXTERNAL_LIBS})
target_link_libraries(neonufft PUBLIC ${NEONUFFT_EXTERNAL_LIBS_PUBLIC})
target_include_directories(neonufft INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
target_include_directories(neonufft INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)
set_target_properties(neonufft PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)

# add alias target
add_library(neonufft::neonufft ALIAS neonufft)

set_target_properties(neonufft
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# include GPU library
if(NEONUFFT_CUDA OR NEONUFFT_ROCM)
	add_subdirectory(gpu)
endif()

# generate export header to control symbol visibility
include(GenerateExportHeader)
generate_export_header(neonufft)
configure_file("${CMAKE_CURRENT_BINARY_DIR}/neonufft_export.h"
	"${PROJECT_BINARY_DIR}/include/neonufft/neonufft_export.h"
	COPYONLY
)

# set packge config names
get_target_property(_LIB_TYPE neonufft TYPE)
if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
	set(NEONUFFT_VERSION_FILE "neonufftStaticConfigVersion.cmake")
	set(NEONUFFT_CONFIG_FILE "neonufftStaticConfig.cmake")
	set(NEONUFFT_TARGETS_FILE "neonufftStaticTargets.cmake")
else()
	set(NEONUFFT_VERSION_FILE "neonufftSharedConfigVersion.cmake")
	set(NEONUFFT_CONFIG_FILE "neonufftSharedConfig.cmake")
	set(NEONUFFT_TARGETS_FILE "neonufftSharedTargets.cmake")
endif()


# generate cmake package
include(CMakePackageConfigHelpers)

set(NEONUFFT_CONFIG_OUT_DIR "${PROJECT_BINARY_DIR}/lib/cmake/neonufft")

write_basic_package_version_file(
	"${NEONUFFT_CONFIG_OUT_DIR}/${NEONUFFT_VERSION_FILE}"
	VERSION ${Upstream_VERSION}
	COMPATIBILITY AnyNewerVersion
)

export(TARGETS neonufft NAMESPACE neonufft:: FILE ${NEONUFFT_CONFIG_OUT_DIR}/${NEONUFFT_TARGETS_FILE})
configure_file(${PROJECT_SOURCE_DIR}/cmake/${NEONUFFT_CONFIG_FILE}
	"${NEONUFFT_CONFIG_OUT_DIR}/${NEONUFFT_CONFIG_FILE}"
	@ONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/neonufftConfig.cmake
	"${NEONUFFT_CONFIG_OUT_DIR}/neonufftConfig.cmake"
	@ONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/neonufftConfigVersion.cmake
	"${NEONUFFT_CONFIG_OUT_DIR}/neonufftConfigVersion.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/neonufftTargets.cmake
	"${NEONUFFT_CONFIG_OUT_DIR}/neonufftTargets.cmake"
	COPYONLY
)

# installation commands
if(NEONUFFT_INSTALL_LIB)
	set(NEONUFFT_TARGETS neonufft)
	if(NEONUFFT_CUDA OR NEONUFFT_ROCM)
	    list(APPEND NEONUFFT_TARGETS neonufft_gpu)
	endif()

	install(TARGETS ${NEONUFFT_TARGETS} DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT neonufftTargets)
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/neonufft/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neonufft PATTERN "gpu" EXCLUDE PATTERN "config.in.h" EXCLUDE)
	if(NEONUFFT_CUDA OR NEONUFFT_ROCM)
	    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/neonufft/gpu/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neonufft/gpu)
	endif()
	install(FILES ${PROJECT_BINARY_DIR}/include/neonufft/config.h ${PROJECT_BINARY_DIR}/include/neonufft/neonufft_export.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neonufft)
	install(EXPORT neonufftTargets NAMESPACE neonufft:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neonufft FILE ${NEONUFFT_TARGETS_FILE})
	install(
	  FILES
	  "${NEONUFFT_CONFIG_OUT_DIR}/neonufftConfig.cmake"
	  "${NEONUFFT_CONFIG_OUT_DIR}/neonufftTargets.cmake"
	  "${NEONUFFT_CONFIG_OUT_DIR}/neonufftConfigVersion.cmake"
	  "${NEONUFFT_CONFIG_OUT_DIR}/${NEONUFFT_CONFIG_FILE}"
	  "${NEONUFFT_CONFIG_OUT_DIR}/${NEONUFFT_VERSION_FILE}"
	  DESTINATION
	      ${CMAKE_INSTALL_LIBDIR}/cmake/neonufft
	)

	install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/neonufft"
        FILES_MATCHING PATTERN "*.cmake")
endif()
