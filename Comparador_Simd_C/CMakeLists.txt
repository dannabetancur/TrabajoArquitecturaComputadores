cmake_minimum_required(VERSION 3.16)
project(ComparadorSimdC C)
set(CMAKE_C_STANDARD 11)

# Optimizaciones y SIMD
add_compile_options(-O3 -march=native -mavx2 -mfma)

# Carpeta de headers
include_directories(include)

add_executable(comparador_simd
    src/main.c
    src/vec_add_scalar.c     # <- este faltaba
    src/vec_add_simd.c
)
set_source_files_properties(src/vec_add_scalar.c
    PROPERTIES COMPILE_FLAGS "-O0"
)