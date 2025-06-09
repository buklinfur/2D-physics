cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(LBMSim LANGUAGES CXX CUDA)

cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE "GLVND")

cmake_policy(SET CMP0146 NEW)
cmake_policy(SET CMP0104 NEW)


set(CUDA_ARCH_LIST "" CACHE STRING
    "Semicolon-separated list of GPU architectures (e.g. \"60;70;80\").\
     If empty, try nvidia-smi, else fall back to '60;70'.")

if (CUDA_ARCH_LIST)
  message(STATUS "Using user-provided CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST} CACHE STRING "" FORCE)
else()
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _cc_str
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if (_cc_str)
    string(REPLACE "." "" _cc_list "${_cc_str}")
    message(STATUS "Detected GPU compute capabilities: ${_cc_list}")
    set(CMAKE_CUDA_ARCHITECTURES ${_cc_list} CACHE STRING "" FORCE)
  else()
    set(_default_arch "60;70")
    message(WARNING "Could not detect GPU. Falling back to default architectures: ${_default_arch}")
    set(CMAKE_CUDA_ARCHITECTURES ${_default_arch} CACHE STRING "" FORCE)
  endif()
endif()


add_subdirectory(CUDAModules)
# add_subdirectory(App)
add_subdirectory(UpdateLoader)

file(GLOB source_files "main.cu")

add_executable(executables  ${source_files})
set_target_properties(executables PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(executables PRIVATE CUDAFacade CUDAGrid CUDADraw Obstacles)
include_directories(include)