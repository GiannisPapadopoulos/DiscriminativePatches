# _SourceFiles.cmake
set( RelativeDir "src/data" )
set( RelativeSourceGroup "Source Files\\data" )

set( DirFiles
	DataSet.cpp
	DataSet.h
	GenerateTestData.cpp
	GenerateTestData.h
	TrainingData.cpp
	TrainingData.h
)

set( DirFiles_SourceGroup "${RelativeSourceGroup}" )
set( LocalSourceGroupFiles )

foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )
