#include "kmeans.h"
#include "float.h"
#include "common.h"
/**
 * This code implements the dynamic programming algorithm
 * for computing k-means.
 */

static double oo = DBL_MAX;
static struct IntervalSum ps;
static struct Table t;

static void fill_row(size_t k) {
    // fill row k, i.e. k clusters.
    // We know k >= 2.
    size_t n = t.n;

    // i is the last point of the last cluster.
    for (size_t i = 1; i < n; ++i) {
        double best = oo;
        // j is the first point of the last cluster.
        for (size_t j = 1; j <= i; ++j) {
            double cost = cost_interval_l2(&ps, j, i) + get(&t, k - 1, j - 1);
            if (cost < best) {
                best = cost;
            }
        }
        set(&t, k, i, best);
    }
    return;
}

static void base_case(size_t k) {
    size_t n = t.n;
    for (size_t i = 0; i < n; ++i) {
        double cost = cost_interval_l2(&ps, 0, i);
        set(&t, 1, i, cost);
    }
    set(&t, 1, 0, 0.0);

    for (size_t j = 1; j <= k; ++j) {
        set(&t, j, 0, 0.0);
    }
    return;
}


// static function not visible in other compilation units.
static double kmeans(double *points, size_t n,
                     double *centers, size_t k) {
    init_Table(&t, k + 1, n, 0.0);
    init_IntervalSum(&ps, points, n);
    base_case(k);
    evict_row(&t, 0);
    for (size_t c = 2; c <= k; ++c) {
        fill_row(c);
        evict_row(&t, c-1);
    }
    double ret = get(&t, k, n - 1);
    free_Table(&t);
    free_IntervalSum(&ps);
    return ret;
}


kmeans_fn get_kmeans_slow() {
    return kmeans;
}
