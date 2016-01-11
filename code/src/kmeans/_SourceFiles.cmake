# _SourceFiles.cmake
set( RelativeDir "src/kmeans" )
set( RelativeSourceGroup "Source Files\\kmeans" )

set( DirFiles
	umKmeans.cpp
	umKmeans.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
