#include "kmeans.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
/**
 * This code implements the dynamic programming algorithm
 * for computing k-means.
 */

kmeans_slow::kmeans_slow(const std::vector<double> &points) : kmeans_dp(points) {}
std::string kmeans_slow::name() { return std::string("slow"); }
std::unique_ptr<kmeans_dp> kmeans_slow::get_instance(std::vector<double> &points) {
    return std::unique_ptr<kmeans_dp>(new kmeans_slow(points));
}

std::unique_ptr<kmeans_result> kmeans_slow::compute(size_t k) {
    std::unique_ptr<kmeans_result> res = std::unique_ptr<kmeans_result>(new kmeans_result);
    row = std::vector<double>(n, 0);
    row_prev = std::vector<double>(n, 0);
    for (size_t i = 1; i < n; ++i) {
        double cost = is.cost_interval_l2(0, i);
        row[i] = cost;
    }

    for (size_t _k = 2; _k <= k; ++_k) {
        std::swap(row, row_prev);
        for (size_t i = 0; i < n; ++i) {
            row[i] = std::numeric_limits<double>::max();
            if (i < _k) {
                row[i] = 0;
                continue;
            }
            for (size_t j = 1; j <= i; ++j) {
                double cost = is.cost_interval_l2(j, i) + row_prev[j-1];
                row[i] = std::min(row[i], cost);
            }
        }
    }
    res->cost = row[n - 1];
    return res;
}

static double kmeans_object_oriented(double *points_ptr, size_t n,
                                     double *centers_ptr, size_t k) {
    std::vector<double> points(n, 0);
    for (size_t i = 0; i < n; ++i) points[i] = points_ptr[i];
    kmeans_slow kmeans_slow(points);
    std::unique_ptr<kmeans_result> kmeans_res = kmeans_slow.compute(k);
    if (centers_ptr) {
        for (size_t i = 0; i < k; ++i) centers_ptr[i] = kmeans_res->centers[i];
    }
    return kmeans_res->cost;
}

kmeans_fn get_kmeans_slow() {
    return kmeans_object_oriented;
}
