set(PROJECTNAME "SwcLoader")

if(WIN32 )
 
elseif(UNIX)

	set(CMAKE_BUILD_TYPE Release)

endif()

FILE(GLOB sourcefile ./*)

add_library(${PROJECTNAME} SHARED ${sourcefile} )

if(WIN32 )
 
elseif(UNIX)

endif()


