# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
project(HelloWorld)
find_package(CUDA QUIET REQUIRED)
# Specify binary name and source file to build it from
cuda_add_executable(
    HelloWorld
    hellocuda.cu)