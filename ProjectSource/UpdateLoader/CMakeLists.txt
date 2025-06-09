project(UpdateLoader LANGUAGES CXX)

add_library(UpdateLoader STATIC
    UpdateLoader.cpp
    UpdateLoader.hpp
)

find_package(CURL REQUIRED)
target_include_directories(UpdateLoader PUBLIC 
    include 
    ${CURL_INCLUDE_DIRS}
)
target_link_libraries(UpdateLoader PRIVATE ${CURL_LIBRARIES})

file(COPY scripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})