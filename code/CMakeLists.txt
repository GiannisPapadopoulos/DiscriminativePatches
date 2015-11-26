# CMakeLists.txt

cmake_minimum_required(VERSION 2.8)

if(UNIX)
    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

project( MAI_project4 )

#SET(OpenCV_DIR "/home/stefan/Downloads/opencv-2.4.11/install/share/OpenCV")
SET(OpenCV_DIR "/home/giannis/Downloads/large/opencv-2.4.11/install")
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIR} )

#SET(Boost_INCLUDE_DIR /home/stefan/Downloads/boost_1_59_0/install/include) 
SET(Boost_INCLUDE_DIR /home/giannis/Downloads/large/boost_1_59_0) 


find_package( Boost COMPONENTS system filesystem REQUIRED)

include_directories( ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

include( "src/_SourceFiles.cmake" )

add_executable( ${PROJECT_NAME} ${ProjectSources} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} )

# Build command unix: cmake -G "Eclipse CDT4 - Unix Makefiles" .
