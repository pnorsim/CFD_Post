cmake_minimum_required(VERSION 3.18)
project(cfdpost LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CFDPOST_PYTHON "Build the Python extension module" ON)
option(CFDPOST_NATIVE "Optimise for the build machine (-march=native)" OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(ZLIB)

add_library(cfdpost_core STATIC
  cpp/src/mmap_file.cpp
  cpp/src/comm.cpp
  cpp/src/vtu.cpp
  cpp/src/su2.cpp
  cpp/src/cells.cpp
  cpp/src/partition.cpp
  cpp/src/contour.cpp
  cpp/src/surface.cpp)
target_include_directories(cfdpost_core PUBLIC cpp/include)
target_link_libraries(cfdpost_core PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(cfdpost_core PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
# only the MPI C API is used: skip the deprecated C++ bindings
target_compile_definitions(cfdpost_core PUBLIC OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
if(CFDPOST_NATIVE)
  target_compile_options(cfdpost_core PRIVATE -march=native)
endif()
if(ZLIB_FOUND)
  target_link_libraries(cfdpost_core PUBLIC ZLIB::ZLIB)
  target_compile_definitions(cfdpost_core PUBLIC CFDPOST_HAVE_ZLIB)
endif()

if(CFDPOST_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  if(NOT pybind11_DIR)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core cpp/python/module.cpp)
  target_link_libraries(_core PRIVATE cfdpost_core)
  # in-tree build drops the module next to the Python sources (editable workflow)
  if(NOT CFDPOST_MODULE_DIR)
    set(CFDPOST_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/cfdpost)
  endif()
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CFDPOST_MODULE_DIR})
endif()
