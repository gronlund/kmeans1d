#include "common.h"
#include "stdlib.h"
#include "assert.h"
#include "stdio.h"

void init_Table(struct Table *t, size_t k, size_t n, double init_val) {
    t->n = n;
    t->k = k;
    t->arr = (double*) malloc(n*k*sizeof(double));
    for (size_t i = 0; i < n*k; ++i) {
        t->arr[i] = init_val;
    }
}

double get(struct Table *t, size_t i, size_t j) {
    size_t n = t->n;
    size_t idx = i*n+j;
    return t->arr[idx];
}

void set(struct Table *t, size_t i, size_t j, double val) {
    size_t n = t->n;
    size_t k = t->k;
    size_t idx = i*n+j;
    assert(idx < k*n);
    t->arr[i*n + j] = val;
}

void free_Table(struct Table *t) {
    free(t->arr);
}

void print_Table(struct Table *t) {
    size_t k = t->k;
    size_t n = t->n;
    for (size_t i = 0; i < k; ++i) {
        for (size_t m = 0; m < n; ++m) {
            printf("%.2f ", get(t, i, m));
        }
        printf("\n");
    }
}

void evict_row(struct Table *t, size_t row) {
    return;
}
