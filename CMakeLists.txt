cmake_minimum_required(VERSION 3.10)
project(project_empire CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(project_empire main.cu heap.h common.h)

set_target_properties(project_empire PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
