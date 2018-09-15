cmake_minimum_required(VERSION 2.8.10)

project(bp)
# Common files
include_directories(${PROJECT_SOURCE_DIR}/include)
SET(SOURCES
  src/main.cpp
  src/iris.cpp
  src/bp.cpp
)

SET(HEADERS
 include/iris.hpp
 include/bp.hpp
)
add_executable(bp      ${SOURCES}
      ${HEADERS}   # need this only to find header files in Visual Studio
    )