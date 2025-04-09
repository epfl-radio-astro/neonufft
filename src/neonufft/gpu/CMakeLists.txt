set(NEONUFFT_GPU_SOURCE_FILES
	plan.cpp
	device_allocator.cpp
	util/fft_grid.cpp
	)
set(NEONUFFT_GPU_KERNEL_FILES
	kernels/upsample_kernel.cu
	kernels/interpolation_kernel.cu
	kernels/rescale_loc_kernel.cu
	kernels/spreading_kernel.cu
	kernels/downsample_kernel.cu
	kernels/min_max_kernel.cu
	kernels/prephase_kernel.cu
	kernels/postphase_kernel.cu
	kernels/fseries_kernel.cu
	)

if(NEONUFFT_ROCM)
	set_source_files_properties(${NEONUFFT_GPU_KERNEL_FILES} PROPERTIES LANGUAGE HIP)
endif()

add_library(neonufft_gpu_objects OBJECT ${NEONUFFT_GPU_SOURCE_FILES} ${NEONUFFT_GPU_KERNEL_FILES})

# make static libraries linkable into shared ones
set_property(TARGET neonufft_gpu_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(neonufft_gpu_objects PRIVATE ${NEONUFFT_FLAGS})
target_include_directories(neonufft_gpu_objects PRIVATE ${NEONUFFT_INCLUDE_DIRS})
target_link_libraries(neonufft_gpu_objects PRIVATE ${NEONUFFT_GPU_EXTERNAL_LIBS})
target_link_libraries(neonufft_gpu_objects PUBLIC ${NEONUFFT_GPU_EXTERNAL_LIBS_PUBLIC})
target_include_directories(neonufft_gpu_objects INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
# set_target_properties(neonufft_gpu_objects PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)
set_property(TARGET neonufft_gpu_objects PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
set_property(TARGET neonufft_gpu_objects PROPERTY CUDA_SEPARABLE_COMPILATION OFF)


#shared library
add_library(neonufft_gpu $<TARGET_OBJECTS:neonufft_gpu_objects>)

set_property(TARGET neonufft_gpu PROPERTY VERSION ${NEONUFFT_VERSION})
set_property(TARGET neonufft_gpu PROPERTY SOVERSION ${NEONUFFT_SO_VERSION})
set_property(TARGET neonufft_gpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
set_property(TARGET neonufft_gpu PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_link_libraries(neonufft_gpu PRIVATE ${NEONUFFT_GPU_EXTERNAL_LIBS})
target_link_libraries(neonufft_gpu PUBLIC ${NEONUFFT_GPU_EXTERNAL_LIBS_PUBLIC} neonufft::neonufft)
target_include_directories(neonufft_gpu INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
target_include_directories(neonufft_gpu INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)
# set_target_properties(neonufft_gpu PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)

# add alias target
add_library(neonufft::neonufft_gpu ALIAS neonufft_gpu)


set_target_properties(neonufft_gpu
    PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

