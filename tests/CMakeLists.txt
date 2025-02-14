set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)

if(NEONUFFT_BUNDLED_GOOGLETEST)
  # add googletest
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_MD5 7e11f6cfcf6498324ac82d567dcb891e
  )
  FetchContent_MakeAvailable(googletest)
else()
  find_package(googletest CONFIG REQUIRED)
endif()

# test executables
add_executable(run_tests
  run_tests.cpp
  test_prephase.cpp
  test_postphase.cpp
  test_type_1.cpp
  test_type_2.cpp
  test_type_3.cpp
)
target_link_libraries(run_tests PRIVATE gtest $<TARGET_OBJECTS:neonufft_objects> $<TARGET_OBJECTS:neonufft_contrib_objects> ${NEONUFFT_EXTERNAL_LIBS} ${NEONUFFT_EXTERNAL_LIBS_PUBLIC})
target_include_directories(run_tests PRIVATE ${NEONUFFT_INCLUDE_DIRS})

set_target_properties(run_tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

