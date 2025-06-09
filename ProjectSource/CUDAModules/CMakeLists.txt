cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDAModules LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

add_library(CUDAGrid STATIC
    CUDAGrid/CUDAGrid.cu
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

add_library(Obstacles STATIC
    Obstacles/Obstacles.cu
    Obstacles/Obstacles.cuh
    Obstacles/ObstaclesFactory.cu
    Obstacles/ObstaclesFactory.cuh
    )

target_compile_features(Obstacles PUBLIC cxx_std_17)
set_target_properties(Obstacles PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(CUDAFacade STATIC
    CUDAFacade/CUDAFacade.cu
    CUDAFacade/CUDAFacade.cuh
    )

target_compile_features(CUDAFacade PUBLIC cxx_std_17)
set_target_properties(CUDAFacade PROPERTIES CUDA_SEPARABLE_COMPILATION ON)