############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
# Updated 7.15.2020
############################################################################

# collect source files
cmake_minimum_required(VERSION 3.8)
set (CMAKE_CXX_STANDARD 11)

project(HW1 LANGUAGES CUDA CXX)
set (CMAKE_CXX_FLAGS "-O3 -Wall  -Wextra -m64")
set (CMAKE_CUDA_FLAGS "-O3 -fmad=false -Xcompiler -Wall -Xcompiler -Wextra -m64")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
find_package(OpenCV REQUIRED)

SET (HW1_files main.cpp reference_calc.cpp compare.cpp)

ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
target_link_libraries(HW1 ${OpenCV_LIBS})
