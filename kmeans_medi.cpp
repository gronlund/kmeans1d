#include "kmeans.h"
#include "float.h"
#include "common.h"

#include <limits>
/**
 * This code implements the divide-and-conquer algorithm
 * for computing k-means.
 */

static double oo = DBL_MAX;
static struct IntervalSum ps;
static struct Table t;


kmeans_medi::kmeans_medi(const std::vector<double> &points) : is(points), n(points.size()) { }

void kmeans_medi::fill_row_rec(size_t begin, size_t end, size_t k,
                               int64_t split_left, int64_t split_right) {

    size_t mid = (begin+end)/2;

    double best = std::numeric_limits<double>::max();
    int64_t best_split = mid;
    for (int64_t s = split_left; s <= split_right && s <= mid; ++s) {
        double cost_last_cluster = is.cost_interval_l2(s, mid);
        double cost_before = row_prev[s-1];
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

    row[mid] = best;

    if (mid != begin) {
        fill_row_rec(begin, mid, k, split_left, best_split);
    }
    if (mid + 1 != end) {
        fill_row_rec(mid+1, end, k, best_split, split_right);
    }
}

void kmeans_medi::fill_row(size_t k) {
    fill_row_rec(0, n, k, 0, n-1);
}

std::unique_ptr<kmeans_result> kmeans_medi::compute(size_t k) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);
    row = std::vector<double>(n, 0.0);
    row_prev = std::vector<double>(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        row[i] = is.cost_interval_l2(0, i);
    }

    for (size_t _k = 2; k <= k; ++k) {
        std::swap(row, row_prev);
        fill_row(_k);
    }
    res->cost = row[n-1];
    return res;
}

static double kmeans_object_oriented(double *points_ptr, size_t n,
                                     double *centers_ptr, size_t k) {
    std::vector<double> points(n, 0);
    for (size_t i = 0; i < n; ++i) points[i] = points_ptr[i];
    kmeans_hirschberg_larmore kmeans_hirsch(points);
    std::unique_ptr<kmeans_result> kmeans_res = kmeans_hirsch.compute(k);
    if (centers_ptr) {
        for (size_t i = 0; i < k; ++i) centers_ptr[i] = kmeans_res->centers[i];
    }
    return kmeans_res->cost;
}


kmeans_fn get_kmeans_medi() {
    return kmeans_object_oriented;
}
