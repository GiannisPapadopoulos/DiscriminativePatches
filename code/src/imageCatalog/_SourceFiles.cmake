# _SourceFiles.cmake
set( RelativeDir "src/imageCatalog" )
set( RelativeSourceGroup "Source Files\\imageCatalog" )

set( DirFiles
	CatalogTraining.cpp
	CatalogTraining.h
	CatalogClassificationSVM.cpp
	CatalogClassificationSVM.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
