cmake_minimum_required(VERSION 3.0)
project(Plugins)
if(WIN32 )
 
elseif(UNIX)

	set(CMAKE_BUILD_TYPE Release)

endif()

SET(CMAKE_CXX_FLAGS "-stdlib=libstdc++ -std=c++11")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build/install)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build/install)
SET(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/build/install)

add_subdirectory(SwcLoader)
add_subdirectory(demo)

