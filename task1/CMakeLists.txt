cmake_minimum_required(VERSION 3.6)
project(hpc_pollard)
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-lntl")

find_package(Boost)
set (Boost_USE_MULTITHREAD ON)  # enable multithreading
IF (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
endif()

set(SOURCE_FILES pollard_rho.cpp)
add_executable(hpc_pollard ${SOURCE_FILES})