cmake_minimum_required(VERSION 3.6)
project(hpc_pohlig)
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-lntl -fopenmp")

set(SOURCE_FILES pollard_lambda.cpp)
add_executable(hpc_pohlig ${SOURCE_FILES})
