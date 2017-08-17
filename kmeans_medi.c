#include "kmeans.h"
#include "float.h"
#include "common.h"
/**
 * This code implements the divide-and-conquer algorithm
 * for computing k-means.
 */

static double oo = DBL_MAX;
static struct IntervalSum ps;
static struct Table t;


static void fill_row_rec(size_t n, size_t begin, size_t end, size_t k,
                         int64_t split_left, int64_t split_right) {

    size_t mid = (begin+end)/2;

    double best = oo;
    int64_t best_split = mid;
    for (int64_t s = split_left; s <= split_right && s <= mid; ++s) {
        double cost_last_cluster = cost_interval_l2(&ps, s, mid);
        double cost_before = get(&t, k-1, s-1);
        double cost;
        if (cost_before == oo || cost_last_cluster == oo) {
            cost = oo;
        } else {
            cost = cost_before + cost_last_cluster;
        }
        if (cost < best) {
            best = cost;
            best_split = s;
        }
    }

    set(&t, k, mid, best);

    if (mid != begin) {
        fill_row_rec(n, begin, mid, k, split_left, best_split);
    }
    if (mid + 1 != end) {
        fill_row_rec(n, mid+1, end, k, best_split, split_right);
    }

    return;
}

static void fill_row(size_t k) {
    size_t n = t.n;
    fill_row_rec(n, 0, n, k, 0, n-1);
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

static double kmeans(double *points, size_t n,
                     double *last_row, size_t k) {
    init_Table(&t, k + 1, n, 0.0);
    init_IntervalSum(&ps, points, n);
    base_case(k);
    evict_row(&t, 0);
    for (size_t c = 2; c <= k; ++c) {
        fill_row(c);
        evict_row(&t, c-1);
    }
    double ret = get(&t, k, n - 1);
    if (last_row != 0) {
        for (size_t i = 0; i < n; ++i) last_row[i] = get(&t, k, i);
    }
    free_Table(&t);
    free_IntervalSum(&ps);
    return ret;
}


kmeans_fn get_kmeans_medi() {
    return kmeans;
}
