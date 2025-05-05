set(NEONUFFT_CONTRIB_SOURCE_FILES
	legendre_rule/legendre_rule_fast.cpp
	es_kernel/es_kernel_horner_coeff.cpp
	es_kernel/es_kernel_horner_coeff_125.cpp
	)


add_library(neonufft_contrib_objects OBJECT ${NEONUFFT_CONTRIB_SOURCE_FILES})

# make static libraries linkable into shared ones
set_property(TARGET neonufft_contrib_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(neonufft_contrib_objects PRIVATE ${NEONUFFT_FLAGS})
target_include_directories(neonufft_contrib_objects PRIVATE ${NEONUFFT_INCLUDE_DIRS})
target_link_libraries(neonufft_contrib_objects PRIVATE ${NEONUFFT_EXTERNAL_LIBS})
target_link_libraries(neonufft_contrib_objects PRIVATE ${NEONUFFT_EXTERNAL_LIBS_PUBLIC})
target_include_directories(neonufft_contrib_objects INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
# set_target_properties(neonufft_contrib_objects PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)
