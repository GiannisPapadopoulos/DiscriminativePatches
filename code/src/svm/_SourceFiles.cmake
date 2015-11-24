# _SourceFiles.cmake
set( RelativeDir "src/svm" )
set( RelativeSourceGroup "Source Files\\svm" )

set( DirFiles
	umSVM.cpp
	umSVM.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
