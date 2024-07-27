cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(skeleton-stencil-distributed ${SrcFiles})

target_link_libraries(skeleton-stencil-distributed 
	PUBLIC libNeonSkeleton
	PUBLIC gtest_main
	PRIVATE ${MPI_CXX_LIBRARIES}
	PRIVATE ${NCCL_DIR}/lib/libnccl.so) # TODO: Ted: Confirm the usage of <PRIVATE/PUBLIC/INTERFACE>. # TODO: Ted: Also check with Max about the <libnccl.so> or <libnccl.so.2>.

set_target_properties(skeleton-stencil-distributed PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(skeleton-stencil-distributed PROPERTIES FOLDER "libNeonSkeletonTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "skeleton-stencil-distributed" FILES ${SrcFiles})

add_test(NAME skeleton-stencil-distributed COMMAND skeleton-stencil-distributed)
