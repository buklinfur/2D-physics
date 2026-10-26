project(UpdateLoader LANGUAGES CXX)

# GitHub "owner/name" the self-updater queries. Change it if the repository is renamed or forked.
set(LBMSIM_UPDATE_REPO "buklinfur/2D-physics" CACHE STRING "GitHub repository used for self-update")

add_library(UpdateLoader STATIC
    UpdateLoader.cpp
    UpdateLoader.hpp
)

find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

target_compile_features(UpdateLoader PUBLIC cxx_std_17)
target_compile_definitions(UpdateLoader PRIVATE LBMSIM_UPDATE_REPO="${LBMSIM_UPDATE_REPO}")
target_include_directories(UpdateLoader PUBLIC ${CURL_INCLUDE_DIRS})
target_link_libraries(UpdateLoader PRIVATE ${CURL_LIBRARIES} OpenSSL::Crypto)

file(COPY scripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
