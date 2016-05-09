############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
# minimum required cmake version
cmake_minimum_required(VERSION 2.8)

project(HW3)
find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)

SET (compare_files compare.cpp)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)
    
CUDA_ADD_EXECUTABLE(HW3 ${HW3_files} ${hdr} ${cu})
target_link_libraries(HW3 ${OpenCV_LIBS})
#add open cv directory.
include_directories(${OpenCV_INCLUDE_DIRS})
