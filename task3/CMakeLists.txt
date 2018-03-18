cmake_minimum_required(VERSION 3.6)
project(hpc_pollard)
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-lntl -fopenmp")

set(SOURCE_FILES pollard_rho.cpp)
add_executable(hpc_pollard ${SOURCE_FILES})