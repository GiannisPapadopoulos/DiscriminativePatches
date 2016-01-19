# _SourceFiles.cmake
set( RelativeDir "src/imageCataloge" )
set( RelativeSourceGroup "Source Files\\imageCataloge" )

set( DirFiles
	CatalogeTraining.cpp
	CatalogeTraining.h
	CatalogeClassificationSVM.cpp
	CatalogeClassificationSVM.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
