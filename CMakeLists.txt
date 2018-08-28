cmake_minimum_required(VERSION 3.1)
project (canonical)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Wno-deprecated-declarations -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE")

FIND_PACKAGE(MPFR)
if(MPFR_FOUND)
	message("MPFR FOUND")
	include_directories(${MPFR_INCLUDES})
endif()

# Catch2 unit testing library
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

#set header files
include_directories(inc)
link_libraries(${MPFR_LIBRARIES})

#set common source files
set (SRCS ${SRCS} src/4d.cpp src/su3.cpp src/gauge.cpp src/dirac_op.cpp src/io.cpp)

# Make unit test executable
set (TEST_SRCS ${SRCS} test/test_su3.cpp test/test_dirac_op.cpp)

add_executable(tests test/main.cpp ${TEST_SRCS})
target_link_libraries(tests Catch)

# main executable
add_executable(evals evals.cpp ${SRCS})
add_executable(polyakov polyakov.cpp ${SRCS})

#enable_testing()
add_test(NAME Addition COMMAND tests)

#Make running the unit tests part of the build, so failed test = failed build
add_custom_command(TARGET tests POST_BUILD COMMAND tests -d yes)
