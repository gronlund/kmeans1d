#include "common.h"
#include "stdlib.h"
#include "assert.h"
#include "stdio.h"
#include <chrono>
#include <ctime>
#include <random>
#include <iostream>

// this seems verbose:
//std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())
// this is better: std::time(nullptr)
static std::mt19937_64 mt(std::time(nullptr));

void set_seed(std::mt19937::result_type val) {
    mt.seed(val);
    return;
}

std::mt19937::result_type random_value() {
    auto val = mt();
    return val;
}

void init_Table(struct Table *t, size_t k, size_t n, double init_val) {
    t->n = n;
    t->k = k;
    t->arr = (double*) malloc(n*2*sizeof(double));
    for (size_t i = 0; i < n*2; ++i) {
        t->arr[i] = init_val;
    }
}

double get(struct Table *t, size_t i, size_t j) {
    return t->arr[j];
}

void set(struct Table *t, size_t i, size_t j, double val) {
    size_t n = t->n;
    t->arr[n + j] = val;
}

void free_Table(struct Table *t) {
    free(t->arr);
}

void print_Table(struct Table *t) {
    size_t n = t->n;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t m = 0; m < n; ++m) {
            printf("%.2f ", t->arr[i*n+m]);
        }
        printf("\n");
    }
}

void evict_row(struct Table *t, size_t row) {
    size_t n = t->n;
    for (size_t i = 0; i < n; ++i) {
        t->arr[i] = t->arr[n+i];
    }
}

void init_IntervalSum(struct IntervalSum *s, const double *points, size_t n) {
    s->prefix_sum = (double*) calloc(n+1, sizeof(double));
    s->prefix_sum_of_squares = (double*) calloc(n+1, sizeof(double));
    double prefix_sum_val = 0;
    double prefix_square_val = 0;
    for (size_t i = 1; i <= n; ++i) {
        prefix_sum_val += points[i-1];
        prefix_square_val += points[i-1]*points[i-1];
        s->prefix_sum[i] = prefix_sum_val;
        s->prefix_sum_of_squares[i] = prefix_square_val;
    }
}

/**
 * @returns sum from i to j of points[l]^2 where i included, j not included.
 */
double query_sq(struct IntervalSum *s, size_t i, size_t j) {
    return s->prefix_sum_of_squares[j] - s->prefix_sum_of_squares[i];
}

/**
 * @returns sum from i to j of points[l] where i included, j not included.
 */
double query(struct IntervalSum *s, size_t i, size_t j) {
    return s->prefix_sum[j] - s->prefix_sum[i];
}

double cost_interval_l2(struct IntervalSum *s,
                               int64_t start, size_t i) {
    double suffix_sum = query(s, start, i+1);
    double suffix_sq = query_sq(s, start, i+1);
    size_t length = i-start+1;
    double mean = suffix_sum / length;
    double interval_cost = suffix_sq + mean*mean*length - 2 * suffix_sum * mean;
    return interval_cost;
}

void free_IntervalSum(struct IntervalSum *s) {
    free(s->prefix_sum);
    free(s->prefix_sum_of_squares);
}

double cost_l2(double *points, size_t n) {
    double mean_val = mean(points, n);
    double cost = 0;
    for (size_t i = 0; i < n; ++i) {
        cost += ((mean_val - points[i]) * (mean_val - points[i]));
    }
    return cost;
}

double mean(double *points, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += points[i];
    }
    double ret = sum / n;
    return ret;
}
