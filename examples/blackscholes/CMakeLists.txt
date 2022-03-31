cmake_minimum_required(VERSION 3.14)

#####################################################################
project(blackscholes VERSION 1.0 LANGUAGES CXX)
#####################################################################

include(cmake/CPM.cmake)
CPMAddPackage("gh:frobnitzem/Hemi@2.0.3")
#CPMAddPackage(
#  NAME Hemi
#  VERSION 2.0.2
#  GITHUB_REPOSITORY frobnitzem/hemi
#)
find_package(Hemi)
message(STATUS "Configured Hemi with GPU = ${HEMI_ENABLE_GPU} (${HEMI_GPU_LANG})")

# Define the source to compile:
add_executable(blackscholes blackscholes.cpp)
HemiLink(blackscholes)

install(TARGETS blackscholes DESTINATION bin)
