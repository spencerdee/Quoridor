cmake_minimum_required(VERSION 3.5)

include(cmake/globber.cmake)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(ProjectName VERSION 1.0 LANGUAGES CXX)

# set(PATTERNS "${PROJECT_SOURCE_DIR}/*.cpp" "${PROJECT_SOURCE_DIR}/*.h")
# file(GLOB_RECURSE PROJECT_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

set(PATTERNS "*.cpp" "*.h" "*.hpp")
globber("${PATTERNS}")

# add_library(LibraryName LIBRARY_SOURCES_HERE)

# Adding something we can run - Output name matches target name
add_executable(ProjectName ${SOURCE_FILES})

# find target directories for include paths
target_include_directories(ProjectName PUBLIC
    ${CMAKE_SOURCE_DIR}/src/common
)

# tell CMake where to find googletest
file(TO_CMAKE_PATH "$ENV{HOME}" HOME_CMAKE)

# find packages
find_package(Threads REQUIRED)

target_link_libraries(ProjectName Threads::Threads ws2_32)