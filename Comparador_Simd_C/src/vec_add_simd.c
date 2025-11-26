#include "vec_add.h"
#include <immintrin.h>  // intrínsecos AVX2

void vec_add_simd(const float *a, const float *b, float *c, size_t n) {
    size_t i = 0;
    size_t step = 8; // 8 floats por vector (256 bits)

    for (; i + step <= n; i += step) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
