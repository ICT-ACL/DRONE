cmake_minimum_required(VERSION 3.10)
project(DRONE CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_library(GRAPH_Lib graph.cu)

add_library(SSSP_Lib algorithm/sssp.cu)
target_link_libraries(SSSP_Lib nccl)

add_library(CC_Lib algorithm/cc.cu)
target_link_libraries(CC_Lib nccl)

add_library(PR_Lib algorithm/pr.cu)
target_link_libraries(PR_Lib nccl)

add_library(ALG_COMMON_Lib algorithm/common.cu)
#target_link_libraries(ALG_COMMON_Lib nccl)

add_executable(DRONE_COMM nccl_test.cu)
target_link_libraries(DRONE_COMM nccl)


#set_target_properties(DRONE_COMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
