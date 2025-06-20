# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(SimpleTubeSim_CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Eigen Dependency ---
# Assumes Eigen is in a standard path or a 'lib' subfolder.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- Define Executable ---
add_executable(SimpleTubeSim main.cpp)

# Link Eigen
target_include_directories(SimpleTubeSim PRIVATE ${EIGEN3_INCLUDE_DIR})

# --- Compiler Flags for Performance ---
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(SimpleTubeSim PRIVATE -O3 -DNDEBUG -march=native)
elseif(MSVC)
    target_compile_options(SimpleTubeSim PRIVATE /O2 /DNDEBUG)
endif()
