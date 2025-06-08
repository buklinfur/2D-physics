cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(LBMSim LANGUAGES CXX CUDA)

cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE "GLVND")

cmake_policy(SET CMP0104 NEW)

execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE CC OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "." "" CC "${CC}")
set(CMAKE_CUDA_ARCHITECTURES ${CC})

add_subdirectory(CUDAModules)
# add_subdirectory(AppFacade)
# add_subdirectory(UpdateLoader)

file(GLOB source_files "main.cu")

add_executable(executables  ${source_files})
set_target_properties(executables PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(executables PRIVATE CUDAFacade CUDAGrid CUDADraw Obstacles)
include_directories(include)