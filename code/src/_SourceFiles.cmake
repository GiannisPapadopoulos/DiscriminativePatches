# _SourceFiles.cmake
set( RelativeDir "src" )
set( RelativeSourceGroup "Source Files" )

set( SubDirs data featureExtraction IO svm utils )

set( DirFiles
	CatalogueDetection.cpp
	CatalogueDetection.h
	Configuration.cpp
	Configuration.h
	Constants.h
	main.cpp
	UDoMLDP.cpp
	UDoMLDP.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )

set( SubDirFiles "" )
foreach( Dir ${SubDirs} )
	list( APPEND SubDirFiles "${RelativeDir}/${Dir}/_SourceFiles.cmake" )
endforeach()

foreach( SubDirFile ${SubDirFiles} )
	include( ${SubDirFile} )
endforeach()

