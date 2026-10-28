cmake_minimum_required(VERSION 3.12)
project(wbc_policy_cpu_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Unoptimized wrapper code would inflate measured latency.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(DEFAULT_ONNXRUNTIME_DIR "${REPO_ROOT}/thirdparty/onnxruntime-linux-x64-1.30.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(DEFAULT_ONNXRUNTIME_DIR "${REPO_ROOT}/thirdparty/onnxruntime-linux-aarch64-1.30.0")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ONNXRUNTIME_DIR "${DEFAULT_ONNXRUNTIME_DIR}" CACHE PATH "ONNX Runtime package root")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h" OR
   NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR "ONNX Runtime missing in ${ONNXRUNTIME_DIR}; run scripts/bootstrap_thirdparty.sh or set ONNXRUNTIME_DIR")
endif()

find_package(Threads REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)

add_library(policy_benchmark_lib STATIC
  options.cpp
  model.cpp
  backends/backend.cpp
  backends/backend_ort.cpp
  backends/backend_onnx.cpp
  backends/backend_openvino.cpp
  backends/backend_openvino_ep.cpp
  benchmark.cpp
  report.cpp
)
target_include_directories(policy_benchmark_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(policy_benchmark_lib PRIVATE
  "${REPO_ROOT}/include"
  "${ONNXRUNTIME_DIR}/include"
)
target_link_libraries(policy_benchmark_lib PRIVATE
  "${ONNXRUNTIME_LIBRARY}"
  openvino::runtime
  Threads::Threads
)
target_compile_definitions(policy_benchmark_lib PUBLIC
  WBC_BENCHMARK_DEFAULT_MODEL="${REPO_ROOT}/config/policy/karate/params/policy.onnx"
)
add_executable(onnx_policy_cpu_benchmark main.cpp)
target_link_libraries(onnx_policy_cpu_benchmark PRIVATE policy_benchmark_lib)
set_target_properties(onnx_policy_cpu_benchmark PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
)
add_executable(policy_output_diff output_diff.cpp)
target_link_libraries(policy_output_diff PRIVATE policy_benchmark_lib)
set_target_properties(policy_output_diff PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
)

include(CTest)
if(BUILD_TESTING)
  add_executable(policy_benchmark_tests tests.cpp)
  target_link_libraries(policy_benchmark_tests PRIVATE policy_benchmark_lib)
  add_test(NAME policy_benchmark_tests COMMAND policy_benchmark_tests)
  set(TEST_MODEL "${REPO_ROOT}/config/policy/karate/params/policy.onnx")
  if(EXISTS "${TEST_MODEL}")
    add_test(NAME policy_benchmark_onnx_cpu COMMAND policy_benchmark_tests onnx "${TEST_MODEL}")
    add_test(NAME policy_benchmark_openvino_cpu COMMAND policy_benchmark_tests openvino "${TEST_MODEL}")
  endif()
endif()