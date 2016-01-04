# _SourceFiles.cmake
set( RelativeDir "src/utils" )
set( RelativeSourceGroup "Source Files\\utils" )

set( DirFiles
	facedetection.cpp
	facedetection.h
	ImageDisplayUtils.h
	ImageDisplayUtils.cpp
)	

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
