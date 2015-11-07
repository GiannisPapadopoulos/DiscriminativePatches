# _SourceFiles.cmake
set( RelativeDir "src/IO" )
set( RelativeSourceGroup "Source Files\\IO" )

set( DirFiles
	IOUtils.cpp
	IOUtils.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
