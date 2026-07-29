# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(SimpleTubeSim_CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Eigen Dependency ---
# Assumes Eigen is in a standard path or a 'lib' subfolder.
# No version pin: CMake treats a two-component version request (e.g. "3.3")
# as an exact-minor-version match, which would reject anything newer than
# 3.3.x (including 3.4 and the 5.x line) even though this code has no
# version-specific dependency.
find_package(Eigen3 REQUIRED NO_MODULE)

# --- Define Executable ---
add_executable(SimpleTubeSim main.cpp)

# Link Eigen
target_link_libraries(SimpleTubeSim PRIVATE Eigen3::Eigen)

# --- Compiler Flags for Performance ---
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(SimpleTubeSim PRIVATE -O3 -DNDEBUG -march=native)
elseif(MSVC)
    target_compile_options(SimpleTubeSim PRIVATE /O2 /DNDEBUG)
endif()
