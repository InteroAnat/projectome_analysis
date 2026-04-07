set(PROJECTNAME "demo")

if(WIN32 )
 
elseif(UNIX)

	set(CMAKE_BUILD_TYPE Release)

endif()

FILE(GLOB sourcefile ./*)

add_executable(${PROJECTNAME} ${sourcefile})
add_dependencies(${PROJECTNAME} SwcLoader)
if(WIN32 )
 TARGET_LINK_LIBRARIES(
	${PROJECTNAME} SwcLoader
	)
elseif(UNIX)
TARGET_LINK_LIBRARIES(
	${PROJECTNAME} SwcLoader
	)
endif()


