#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <malloc.h>     // _aligned_malloc / _aligned_free
#include "vec_add.h"

#define SIZE 10000000   // 10 millones

static double now_seconds(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main(void) {
    // Memoria alineada a 32 bytes (AVX)
    float *a  = (float*)_aligned_malloc(SIZE * sizeof(float), 32);
    float *b  = (float*)_aligned_malloc(SIZE * sizeof(float), 32);
    float *c1 = (float*)_aligned_malloc(SIZE * sizeof(float), 32);
    float *c2 = (float*)_aligned_malloc(SIZE * sizeof(float), 32);

    if (!a || !b || !c1 || !c2) {
        fprintf(stderr, "Error: no se pudo asignar memoria\n");
        if (a)  _aligned_free(a);
        if (b)  _aligned_free(b);
        if (c1) _aligned_free(c1);
        if (c2) _aligned_free(c2);
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < SIZE; ++i) {
        a[i] = (float)(rand() % 100);
        b[i] = (float)(rand() % 100);
    }

    printf("Comparando suma de vectores (%d elementos)\n\n", SIZE);

    double t0 = now_seconds();
    vec_add_scalar(a, b, c1, SIZE);
    double t1 = now_seconds();
    double t_scalar = t1 - t0;
    printf("Scalar: %.4f s\n", t_scalar);

    t0 = now_seconds();
    vec_add_simd(a, b, c2, SIZE);
    t1 = now_seconds();
    double t_simd = t1 - t0;
    printf("SIMD  : %.4f s\n", t_simd);

    int ok = 1;
    for (size_t i = 0; i < SIZE; ++i) {
        if (fabsf(c1[i] - c2[i]) > 1e-5f) { ok = 0; break; }
    }
    printf("\nVerificación: %s\n", ok ? "OK" : "FALLÓ");
    if (t_simd > 0) printf("Speedup SIMD vs Scalar: %.2fx\n", t_scalar / t_simd);

    _aligned_free(a);
    _aligned_free(b);
    _aligned_free(c1);
    _aligned_free(c2);
    return ok ? 0 : 1;
}
