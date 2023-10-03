cmake_minimum_required(VERSION 3.22)
project(pract4_1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
add_executable(pract4_1 main.cpp matrix_op.cpp matrix_op.h)