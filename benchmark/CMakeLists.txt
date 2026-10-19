cmake_minimum_required(VERSION 3.12)
project(wbc_policy_cpu_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(ONNXRUNTIME_DIR "${REPO_ROOT}/thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(ONNXRUNTIME_DIR "${REPO_ROOT}/thirdparty/onnxruntime-linux-aarch64-1.22.0")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h" OR
   NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR "Bundled ONNX Runtime missing; run scripts/bootstrap_thirdparty.sh")
endif()

find_package(Threads REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)

add_executable(onnx_policy_cpu_benchmark main.cpp)
target_include_directories(onnx_policy_cpu_benchmark PRIVATE
  "${REPO_ROOT}/include"
  "${ONNXRUNTIME_DIR}/include"
)
target_link_libraries(onnx_policy_cpu_benchmark PRIVATE
  "${ONNXRUNTIME_LIBRARY}"
  openvino::runtime
  Threads::Threads
)
target_compile_definitions(onnx_policy_cpu_benchmark PRIVATE
  WBC_BENCHMARK_DEFAULT_MODEL="${REPO_ROOT}/config/policy/karate/params/policy.onnx"
)
set_target_properties(onnx_policy_cpu_benchmark PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
)