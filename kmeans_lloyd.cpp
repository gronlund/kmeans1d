#include <algorithm>
#include <ctime>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "kmeans.h"
#include "common.h"

/**
 * This code implements lloyds algorithm using binary searches
 * for computing k-means in 1D.
 */


static double oo = std::numeric_limits<double>::max();
static double eps = 1e-6;

double dist_sq(double p1, double p2) {
    return (p1-p2)*(p1-p2);
}

std::vector<size_t> init_splits(size_t n, size_t k) {
    // resorvoir sample algorithm
    std::vector<size_t> splits(k, 0);
    for (size_t i = 0; i < k-1; ++i) {
        splits[i] = i;
    }

    for (size_t i = k-1; i < n-1; ++i) {
        auto rnd_val = random_value();
        size_t j = rnd_val % i;
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

    IntervalSum is;
    init_IntervalSum(&is, points, n);
    if (k == 1) {
        double cost = cost_interval_l2(&is, 0, n-1);
        free_IntervalSum(&is);
        return cost;
    }
    std::vector<size_t> splits = init_splits(n, k); // splits[i] is the index of the
                                               // last point in cluster i.
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

static double kmeans_slow(double *points, size_t n,
                          double *last_row, size_t k) {
    IntervalSum is;
    init_IntervalSum(&is, points, n);
    if (k == 1) {
        double cost = cost_interval_l2(&is, 0, n-1);
        free_IntervalSum(&is);
        return cost;
    }
    std::vector<size_t> splits = init_splits(n, k); // splits[i] is the index of the
                                               // last point in cluster i.

    bool converged = false;
    while (!converged) {
        std::vector<double> new_means;
        size_t start = 0;
        for (size_t end : splits) {
            double mean = ((double) query(&is, start, end + 1)) / (end + 1 - start);
            new_means.push_back(mean);
            start = end + 1;
        }

        std::vector<size_t> assignment(n, 0);

        for (size_t i = 0; i < n; ++i) {
            double closest_dist = oo;
            for (size_t m = 0; m < new_means.size(); ++m) {
                double mean = new_means[m];
                double dist = abs(mean - points[i]);
                if (dist < closest_dist) {
                    assignment[i] = m;
                    closest_dist = dist;
                }
            }
        }
        std::vector<size_t> new_splits;
        for (size_t i = 0; i < n - 1; ++i) {
            if (assignment[i] != assignment[i+1]) {
                new_splits.push_back(i);
            }
        }
        while (new_splits.size() != splits.size()) {
            new_splits.push_back(n-1);
        }

        bool change_detected = false;
        for (size_t i = 0; i < splits.size(); ++i) {
            if (splits[i] != new_splits[i]) {
                change_detected = true;
                break;
            }
        }
        splits = std::move(new_splits);
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

kmeans_fn get_kmeans_lloyd_slow() {
    return kmeans_slow;
}
