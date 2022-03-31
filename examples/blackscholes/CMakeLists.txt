cmake_minimum_required(VERSION 3.14)

#####################################################################
project(blackscholes VERSION 1.0 LANGUAGES CXX)
#####################################################################

include(cmake/CPM.cmake)
CPMAddPackage(
  NAME Hemi
  VERSION 2.0.1
  GITHUB_REPOSITORY harrism/hemi
)

# Define the source to compile:
add_executable(blackscholes blackscholes.cpp)
HemiLink(blackscholes)

install(TARGETS blackscholes DESTINATION bin)
