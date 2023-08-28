cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(skeleton-stencil ${SrcFiles})
# Ted: According to ChatGPT. To link the executable against MPI. Similarly for NCCL.
target_link_libraries(skeleton-stencil 
	PUBLIC libNeonSkeleton
	PUBLIC gtest_main
	PRIVATE ${MPI_CXX_LIBRARIES}
	PRIVATE $ENV{NCCL_HOME}/lib/libnccl.so)

set_target_properties(skeleton-stencil PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(skeleton-stencil PROPERTIES FOLDER "libNeonSkeleton")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "skeleton-stencil" FILES ${SrcFiles})

add_test(NAME skeleton-stencil COMMAND skeleton-stencil)
