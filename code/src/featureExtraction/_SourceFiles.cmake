# _SourceFiles.cmake
set( RelativeDir "src/featureExtraction" )
set( RelativeSourceGroup "Source Files\\featureExtraction" )

set( DirFiles
	cvHOG.cpp
	cvHOG.h
	umPCA.cpp
	umPCA.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
