#include "kmeans.h"
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include "common.h"

using std::vector;

double dist_sq(double p1, double p2) {
    return (p1-p2)*(p1-p2);
}

vector<size_t> init_splits(size_t n, size_t k) {
    // resorvoir sample algorithm
    vector<size_t> splits(k, 0);
    for (size_t i = 0; i < k-1; ++i) {
        splits[i] = i;
    }
    std::mt19937_64 mt(std::time(0));
    for (size_t i = k-1; i < n-1; ++i) {
        size_t j = mt() % i;
        if (j < k-1) {
            splits[j] = i;
        }
    }
    splits[k-1] = n-1;
    std::sort(splits.begin(), splits.end());
    return splits;
}

static double kmeans(double *points, size_t n,
                     double *last_row, size_t k) {

    vector<size_t> splits = init_splits(n, k); // splits[i] is the index of the
                                               // last point in cluster i.
    IntervalSum is;
    init_IntervalSum(&is, points, n);

    double eps = 1e-6;
    bool converged = false;
    while (!converged) {
        size_t first_point = 0;
        bool change_detected = false;
        for (size_t i = 0; i < k-1; ++i) {
            size_t next_first_point = std::min(splits[i]+1, n-1);
            double mean_curr = query(&is, first_point, splits[i]+1) / (splits[i] + 1 - first_point);
            double mean_next = query(&is, next_first_point, splits[i+1]+1) / (splits[i+1]+1 - next_first_point);

            // binary search to update split point.
            size_t hi = n;
            size_t lo = 0;
            while (hi != lo+1) {
                size_t mid = lo + (hi - lo) / 2;
                if (points[mid] <= mean_curr) {
                    lo = mid;
                    continue;
                }
                double dist_curr = dist_sq(points[mid], mean_curr);
                double dist_next = dist_sq(points[mid], mean_next);
                if (dist_curr > dist_next) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            if (splits[i] != lo) {
                change_detected = true;
            }
            splits[i] = lo;
            first_point = next_first_point;
        }
        if (!change_detected) converged = true;
    }
    size_t start = 0;
    double cost = 0;
    for (size_t i = 0; i < k; ++i) {
        cost += cost_interval_l2(&is, start, splits[i]);
        start = splits[i]+1;
    }
    free_IntervalSum(&is);
    return cost;
}

kmeans_fn get_kmeans_lloyd() {
    return kmeans;
}

