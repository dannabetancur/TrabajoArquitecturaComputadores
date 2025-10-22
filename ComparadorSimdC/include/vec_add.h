#pragma once
#include <stddef.h>

void vec_add_scalar(const float *a, const float *b, float *c, size_t n);
void vec_add_simd(const float *a, const float *b, float *c, size_t n);
