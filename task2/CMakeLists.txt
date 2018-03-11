cmake_minimum_required(VERSION 3.9)
project(openmp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(e_hello hello/example_hello.cpp)
add_executable(c_hello hello/c_hello.cpp)
add_executable(e_loop loop/example_loop.cpp)
add_executable(c_loop loop/c_loop.cpp)
add_executable(e_clauses clauses/example_clauses.cpp)
add_executable(e_schedule schedule/schedule.cpp)