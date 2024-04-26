cmake_minimum_required(VERSION 3.10)
project(gcn LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#set(CMAKE_C_COMPILER mpicc)
#set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_NVCC_COMPILER "/usr/local/cuda/bin/nvcc")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

set(MPI_LIBS mpi)

#set(CUDA_HOST_COMPILER "/usr/bin/mpic++")
#set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-11.6/bin/nvcc")

file(GLOB Common src/common/*.cu)
file(GLOB Cuda src/cuda/*.cu)
file(GLOB CommonH src/common/*.cuh)
file(GLOB CudaH src/cuda/*.cuh)

list(APPEND CUDA_NVCC_FLAGS "--keep")
message(STATUS "CUDA_NVCC_FLAGS = " ${CUDA_NVCC_FLAGS})

#add_library(cudaLib ${Cuda} ${CudaH})
#add_library(commonLib ${Common} ${CommonH})
#add_library(seqLib ${Seq} ${SeqH})

#add_library(cudaLib SHARED ${Cuda})
#add_library(commonLib SHARED ${Common})
add_library(cudaLib ${Cuda})
add_library(commonLib ${Common})

include_directories(src/cuda)
include_directories(src/common)
AUX_SOURCE_DIRECTORY(. src)

add_library(GCN_Lib src/gcn.cu)

add_executable(gcn_test src/gcn.cu)
target_link_libraries(gcn_test cudaLib commonLib nccl cublas cusparse ${MPI_LIBS})
