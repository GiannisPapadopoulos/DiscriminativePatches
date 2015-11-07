# CMakeLists.txt

cmake_minimum_required(VERSION 2.8)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

project( MAI_project4 )

SET(OpenCV_DIR "/home/stefan/Downloads/opencv-2.4.11/install/share/OpenCV")
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

SET(Boost_INCLUDE_DIR /home/stefan/Downloads/boost_1_59_0/install/include)
find_package( Boost COMPONENTS system filesystem REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

include( "src/_SourceFiles.cmake" )

add_executable( ${PROJECT_NAME} ${ProjectSources} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
