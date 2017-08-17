#ifndef __COMMON_H__
#define __COMMON_H__

#include "stdlib.h"
#include <random>

void set_seed(std::mt19937::result_type val);
std::mt19937::result_type random_value();

struct Table {
    size_t k, n;
    /**
     * arr will be of length k'*n  -- with k' = (k+1).
     */
    double *arr;
};

void init_Table(struct Table *t, size_t k, size_t n, double init_val);

/**
 * @param i is the number of clusters, and j is the last point in the
 * clustering.
 */
double get(struct Table *t, size_t i, size_t j);

void set(struct Table *t, size_t i, size_t j, double val);

void free_Table(struct Table *t);

void print_Table(struct Table *t);

void evict_row(struct Table *t, size_t row);

struct IntervalSum {
    double *prefix_sum;
    double *prefix_sum_of_squares;
    size_t n;

};

void init_IntervalSum(struct IntervalSum *s, const double *points, size_t n);

/**
 * @returns sum from i to j of points[l]^2 where i included, j not included.
 */
double query_sq(struct IntervalSum *s, size_t i, size_t j);

/**
 * @returns sum from i to j of points[l] where i included, j not included.
 */
double query(struct IntervalSum *s, size_t i, size_t j);

/**
 * returns the cost of clustering points [start, i] into one cluster with L_2
 * norm.
 */
double cost_interval_l2(struct IntervalSum *s,
                        int64_t start, size_t i);

void free_IntervalSum(struct IntervalSum *s);

double cost_l2(double *points, size_t n);

double mean(double *points, size_t n);
#endif /* __COMMON_H__ */
