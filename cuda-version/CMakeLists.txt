cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CMakeCUDAProject LANGUAGES CXX CUDA)



find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)



file(GLOB source_files "src/*.cu")

add_library(CUDAGrid STATIC
    CUDAGrid/cuda_sim_step.cu
    CUDAGrid/CUDAGrid.cuh
    )
target_compile_features(CUDAGrid PUBLIC cxx_std_17)
set_target_properties(CUDAGrid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_library(CUDADraw STATIC
    CUDADraw/CUDADraw.cu
    CUDADraw/CUDADraw.cuh
    )
target_compile_features(CUDADraw PUBLIC cxx_std_17)
set_target_properties(CUDADraw PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CUDADraw PRIVATE OpenGL::GL GLEW::GLEW glfw cudart)

add_executable(executables  ${source_files})

set_target_properties(executables PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(executables PRIVATE CUDAGrid CUDADraw)

include_directories(include)

