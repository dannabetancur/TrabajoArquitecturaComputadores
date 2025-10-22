#include "vec_add.h"

void vec_add_scalar(const float *a, const float *b, float *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
