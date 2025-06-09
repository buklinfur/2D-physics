cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(AppControls LANGUAGES CXX CUDA)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2  # or “latest” tag
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(nlohmann_json REQUIRED)

add_library(AppFacade STATIC
    AppFacade/AppFacade.cu
    Appfacade/AppFacade.cuh
    )

target_compile_features(AppFacade PUBLIC cxx_std_17)
set_target_properties(AppFacade PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(nlohmann_json REQUIRED)

add_library(ConfigParser STATIC
    ConfigParser/ConfigParser.cpp
    ConfigParser/ConfigParser.hpp
    )

target_link_libraries(ConfigParserLib PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(ConfigParser PUBLIC cxx_std_17)
set_target_properties(ConfigParser PROPERTIES CUDA_SEPARABLE_COMPILATION ON)