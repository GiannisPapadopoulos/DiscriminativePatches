# _SourceFiles.cmake
set( RelativeDir "src/svm" )
set( RelativeSourceGroup "Source Files\\svm" )

set( DirFiles
	CatalogueTraining.cpp
	CatalogueTraining.h
	ClassificationSVM.cpp
	ClassificationSVM.h
	svmtest.cpp
	svmtest.h
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
