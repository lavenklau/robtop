cmake_minimum_required(VERSION 3.22)

project(LargeScaleRobTop)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
# set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
# set(CMAKE_IGNORE_PATH "/mnt/d/Program Files/vcpkg/installed/x64-windows/share/gflags")

enable_language(CUDA)

add_definitions(-w)

set(USE_CONDA OFF CACHE BOOL "in conda environment?")
if(USE_CONDA)
set(PACKAGE_PREFIX $ENV{CONDA_PREFIX})
else()
set(PACKAGE_PREFIX "/home/zhd9702/miniconda3/envs/robtop")
endif()
set(TBB_ROOT ${PACKAGE_PREFIX})
set(CMAKE_MODULE_PATH ${PACKAGE_PREFIX})
set(CMAKE_PREFIX_PATH ${PACKAGE_PREFIX})

# disable default path
# set(CMAKE_FIND_USE_PACKAGE_ROOT_PATH FALSE)
# set(CMAKE_FIND_USE_CMAKE_PATH FALSE)
# set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
# set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
# set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY FALSE)
# set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
# set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY FALSE)

#option(DISABLE_CMAKE_SEARCH_PATHS ON)

include(FetchContent)
include(ExternalProject)
# Fetch OpenVDB 8.2
FetchContent_Declare(
  openvdb
  URL https://github.com/AcademySoftwareFoundation/openvdb/archive/refs/tags/v8.2.0.tar.gz
  URL_HASH MD5=2852fe7176071eaa18ab9ccfad5ec403
)
FetchContent_MakeAvailable(openvdb)
message(STATUS "openvdb_SOURCE_DIR = ${openvdb_SOURCE_DIR} ")
message(STATUS "openvdb_BINARY_DIR = ${openvdb_BINARY_DIR} ")

# Fetch trimesh2
FetchContent_Declare(
  trimesh2
  GIT_REPOSITORY https://github.com/jrock08/Trimesh2.git
  GIT_TAG 121d33992f83aeb9ec823936b5136f8bf854d6f0
)
FetchContent_MakeAvailable(trimesh2)
# set(trimesh_src_prefix  ${CMAKE_CURRENT_BINARY_DIR}/trimesh2/src/trimesh2)
# ExternalProject_Add(trimesh2
#     PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trimesh2
#     GIT_REPOSITORY https://github.com/Forceflow/trimesh2.git
#     GIT_TAG 45a78d23942f7d21609895d2ddd9e6043b6b49b3
#     UPDATE_DISCONNECTED true
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND  make -C ${trimesh_src_prefix} 
#     BUILD_IN_SOURCE TRUE
#     INSTALL_COMMAND ""
#     BUILD_BYPRODUCTS ${trimesh_src_prefix}/lib.Linux64/libtrimesh.a
# )
# set(TRIMESH2_INCLUDE_DIR ${trimesh_src_prefix}/include)
# set(TRIMESH2_LIBRARIES ${trimesh_src_prefix}/lib.Linux64/libtrimesh.a)


FetchContent_Declare(
  spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG 8c7242e08f0fb7f6a0022cfe232e3dc5b5bd4eb4
)
FetchContent_MakeAvailable(spectra)

FetchContent_Declare(
  OpenMesh
  GIT_REPOSITORY https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
  GIT_TAG        OpenMesh-8.1
)
FetchContent_MakeAvailable(OpenMesh)

find_package(Eigen3 REQUIRED)
find_package(glm REQUIRED)
find_package(gflags REQUIRED)
find_package(Boost CONFIG REQUIRED)
find_package(CGAL REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)


file(GLOB_RECURSE sources *.cpp *.cu *.cc *.c)
file(GLOB_RECURSE buildCopy build/*.cpp build/*.cu build/*.cc build/*.c)
list(REMOVE_ITEM sources ${buildCopy})

message(STATUS "found .cpp files ${CPP_LIST}")
message(STATUS "found .cu  files ${CU_LIST} ")
message(STATUS "CUDA headers ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ")

include_directories("${CMAKE_SOURCE_DIR}")
# include_directories("${CMAKE_SOURCE_DIR}/generated")
include_directories("${CMAKE_SOURCE_DIR}/voxelizer")
include_directories("${CMAKE_SOURCE_DIR}/grid")
include_directories("${CMAKE_SOURCE_DIR}/mem")
include_directories("${CMAKE_SOURCE_DIR}/optimization")
include_directories("${CMAKE_SOURCE_DIR}/matlab")
include_directories("${CMAKE_SOURCE_DIR}/culib")
include_directories("${CMAKE_SOURCE_DIR}/element")
include_directories("${CMAKE_SOURCE_DIR}/utils")
include_directories("${CMAKE_SOURCE_DIR}/voxelIO")
include_directories("${CMAKE_SOURCE_DIR}/boundary")
include_directories("${CMAKE_SOURCE_DIR}/volumerender/cuda_samples")

add_executable(robtop  ${sources})
set_target_properties(robtop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(robtop PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(robtop PROPERTIES CUDA_ARCHITECTURES "61;75;86")
target_compile_options(robtop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        -dc
        --extended-lambda
        --std=c++17
        --expt-relaxed-constexpr
        --compile
        >)
target_link_libraries(robtop PUBLIC Eigen3::Eigen)
target_include_directories(robtop PUBLIC CGAL::CGAL)
target_include_directories(robtop PUBLIC ${RAPIDJSON_INCLUDE_DIRS})
target_include_directories(robtop PUBLIC ${OPENVDB_INCLUDE_DIRS})
target_include_directories(robtop PUBLIC ${GFLAGS_INCLUDE_DIR})
target_include_directories(robtop PUBLIC ${SPECTRA_INCLUDE_DIR})
target_link_libraries(robtop INTERFACE glm::glm)
target_include_directories(robtop PUBLIC ${openvdb_SOURCE_DIR}/openvdb/openvdb)
target_include_directories(robtop PUBLIC ${openvdb_SOURCE_DIR}/openvdb)
target_include_directories(robtop PUBLIC ${openvdb_BINARY_DIR}/openvdb/openvdb/openvdb)
target_link_libraries(robtop PUBLIC openvdb_shared)
target_link_libraries(robtop PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(robtop PUBLIC OpenMeshCore)
target_link_libraries(robtop PRIVATE ${GFLAGS_LIBRARIES})
target_link_libraries(robtop PRIVATE ${TRIMESH_LIBRARY})
target_link_libraries(robtop PUBLIC Spectra)
target_link_libraries(robtop PUBLIC CUDA::cudart)
target_link_libraries(robtop PUBLIC CUDA::curand)
target_link_libraries(robtop PUBLIC CUDA::cusolver)
target_link_libraries(robtop PUBLIC CUDA::cublas)
target_link_libraries(robtop PUBLIC CUDA::cusparse)
target_link_libraries(robtop PUBLIC CUDA::cuda_driver)
target_include_directories(robtop PUBLIC ${trimesh2_SOURCE_DIR}/include)
target_link_libraries(robtop PUBLIC trimesh)
# add_dependencies(robtop trimesh2)
# target_include_directories(robtop PUBLIC ${TRIMESH2_INCLUDE_DIR})
#target_link_libraries(robtop PUBLIC ${TRIMESH2_LIBRARIES})


