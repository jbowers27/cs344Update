############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
# Updated 7.22.2020
############################################################################

#Set Minimum Required Version and Compiler Standards
cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
project(HW3 LANGUAGES CXX CUDA)

#Set compiler flags
set(CMAKE_CXX_FLAGS "-O3  -Wall -Wextra -m64")
set(CMAKE_CUDA_FLAGS "-O3 -fmad=false -Xcompiler -Wall  -m64")

#Include locations for use of cuda headers with .cpp files
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#Include OpenCV and required libraries
find_package(OpenCV REQUIRED)

#Glob files and generate executable
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

add_executable(HW3 ${HW3_files} ${hdr} ${cu})
target_link_libraries(HW3 ${OpenCV_LIBS})

