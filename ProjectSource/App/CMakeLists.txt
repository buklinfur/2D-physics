cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(AppControls LANGUAGES CXX CUDA)

add_library(AppFacade STATIC
    AppFacade/AppFacade.cu
    Appfacade/AppFacade.cuh
    )

target_compile_features(AppFacade PUBLIC cxx_std_17)
set_target_properties(AppFacade PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(ConfigParser STATIC
    ConfigParser/ConfigParser.cu
    ConfigParser/ConfigParser.cuh
    )

target_compile_features(ConfigParser PUBLIC cxx_std_17)
set_target_properties(ConfigParser PROPERTIES CUDA_SEPARABLE_COMPILATION ON)