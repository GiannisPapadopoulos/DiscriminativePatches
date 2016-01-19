# _SourceFiles.cmake
set( RelativeDir "src/configuration" )
set( RelativeSourceGroup "Source Files\\configuration" )

set( DirFiles
	Configuration.cpp
	Configuration.h
	Constants.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
