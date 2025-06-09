cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDAModules LANGUAGES CXX CUDA)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(UpdateLoaderLib
    UpdateLoader.cpp
    UpdateLoader.h
)
target_link_libraries(UpdateLoaderLib
    PUBLIC CURL::libcurl nlohmann_json::nlohmann_json
)