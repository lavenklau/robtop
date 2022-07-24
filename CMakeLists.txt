cmake_minimum_required(VERSION 3.8)

project(LargeScaleRobTop)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
set(CMAKE_IGNORE_PATH "/mnt/d/Program Files/vcpkg/installed/x64-windows/share/gflags")

message(STATUS "module path ${CMAKE_MODULE_PATH}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen directory ${EIGEN3_INCLUDE_DIRS}")

find_package(glm REQUIRED)
message(STATUS "glm directory ${GLM_INCLUDE_DIRS}")

#find_package(spectra REQUIRED)
set(SPECTRA_INCLUDE_DIR "" CACHE STRING "include directory of Spectra")
set(TRIMESH_INCLUDE_DIR "" CACHE STRING "include directory of trimesh2")
set(TRIMESH_LIBRARY     "" CACHE STRING "library path to trimesh2")

if ("${TRIMESH_LIBRARY}" STREQUAL "")
    message(FATAL_ERROR "You should provide a library file of trimesh2 (TRIMESH_LIBRARY)
    (https://github.com/Forceflow/trimesh2.git)")
endif()

if ("${TRIMESH_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "You should provide a include directory of trimesh2 (TRIMESH_INCLUDE_DIR)
    (https://github.com/Forceflow/trimesh2.git)")
endif()

if ("${SPECTRA_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Missing Spectra library include directory (SPECTRA_INCLUDE_DIR)
    (https://github.com/yixuan/spectra.git) ")
endif()


find_package(gflags REQUIRED)
message(STATUS "gflags dir ${GFLAGS_INCLUDE_DIR}")

find_package(OpenMesh REQUIRED)
message(STATUS "OpenMesh directory ${OPENMESH_INCLUDE_DIR}")

set (OpenVDB_Module_Path "" CACHE PATH "Directory containing FindOpenVDB.cmake")

LIST ( APPEND CMAKE_MODULE_PATH ${OpenVDB_Module_Path})

find_package(OpenVDB REQUIRED)

find_package(Boost REQUIRED)

find_package(CGAL REQUIRED)

find_package(RapidJSON REQUIRED)

find_package(OpenMP REQUIRED)

enable_language(CUDA)

find_package(CUDA REQUIRED)

# Do what the new package does
find_library(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
if (NOT CUDA_DRIVER_LIBRARY)
    # Don't try any stub directories until we have exhausted all other search locations.
    find_library(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
endif ()

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	# set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB CPP_LIST "*.cpp")
file(GLOB CPP_VOX "voxelizer/*.cpp")
list(APPEND CPP_LIST ${CPP_VOX})

#list(REMOVE_ITEM CPP_LIST "volumerender/volume_renderer.cpp")
#list(FILTER CPP_LIST EXCLUDE REGEX ".*volume_renderer\\.cpp$")

file(GLOB CU_LIST "*.cu")
file(GLOB CU_VOX "voxelizer/*.cu")
list(APPEND CU_LIST ${CU_VOX})

message(STATUS "found .cpp files ${CPP_LIST}")
message(STATUS "found .cu  files ${CU_LIST} ")
message(STATUS "CUDA headers ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ")

#include_directories("${OPENMESH_INCLUDE_DIRS}")
include_directories("${TRIMESH_INCLUDE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/generated")
include_directories("${CMAKE_SOURCE_DIR}/voxelizer")
include_directories("${CMAKE_SOURCE_DIR}/volumerender/cuda_samples")
include_directories(${EIGEN3_INCLUDE_DIRS})

add_definitions(-DOM_STATIC_BUILD -D_USE_MATH_DEFINES)

# output include directories
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
	message(STATUS "dir='${dir}'")
endforeach()
add_executable(robtop  ${CPP_LIST})

target_include_directories(robtop PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(cuLib STATIC ${CU_LIST})
set_target_properties(cuLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuLib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#set_target_properties(cuLib PROPERTIES CUDA_ARCHITECTURES all)
# set_target_properties(cuLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(cuLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--relocatable-device-code=true
	--extended-lambda
	--std=c++17
	--expt-relaxed-constexpr
	--compile
	-gencode arch=compute_52,code=sm_52
	>)
target_link_libraries(cuLib PUBLIC glm::glm)
target_link_libraries(cuLib PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(cuLib PRIVATE ${CUDA_DRIVER_LIBRARY})


message(STATUS "CUDA libs ${CUDA_LIBRARIES}")
message(STATUS "CUDA lib  ${CUDA_DRIVER_LIBRARY}")

target_include_directories(robtop PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories(robtop PUBLIC ${GLM_INCLUDE_DIRS})
target_include_directories(robtop PUBLIC ${CGAL_INCLUDE_DIRS})
target_include_directories(robtop PUBLIC ${RAPIDJSON_INCLUDE_DIRS})
target_include_directories(robtop PUBLIC ${OPENVDB_INCLUDE_DIRS})
target_include_directories(robtop PUBLIC ${GFLAGS_INCLUDE_DIR})
target_include_directories(robtop PUBLIC ${SPECTRA_INCLUDE_DIR})
target_link_libraries(robtop INTERFACE glm::glm)
#target_link_libraries(robtop INTERFACE Spectra::Spectra)
target_link_libraries(robtop PRIVATE cuLib)
target_link_libraries(robtop PRIVATE OpenVDB::openvdb)
target_link_libraries(robtop PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(robtop PUBLIC OpenMeshCore)
#target_link_libraries(robtop PRIVATE ${OPENMESH_LIBRARIES})
target_link_libraries(robtop PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(robtop PRIVATE ${CUDA_DRIVER_LIBRARY})
target_link_libraries(robtop PRIVATE ${CUDA_curand_LIBRARY})
target_link_libraries(robtop PRIVATE ${CUDA_cusolver_LIBRARY})
target_link_libraries(robtop PRIVATE ${CUDA_cusparse_LIBRARY})
target_link_libraries(robtop PRIVATE ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(robtop PRIVATE ${GFLAGS_LIBRARIES})
target_link_libraries(robtop PRIVATE ${TRIMESH_LIBRARY})

