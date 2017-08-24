#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <random>
#include <random>
#include <utility>
#include <vector>

#include "kmeans.h"
#include "interval_sum.hpp"
/**
 * This code implements lloyds algorithm using binary searches
 * for computing k-means in 1D.
 */
static double oo = std::numeric_limits<double>::max();
static double eps = 1e-6;

static double dist_sq(double p1, double p2) {
    return std::abs(p1-p2);
}

template<typename T>
static void print_splits(std::vector<T> &splits, std::string prefix) {
    std::cout << "[" << prefix << "] ";
    for (size_t i = 0; i < splits.size(); ++i) {
        std::cout << splits[i] << "   ";
    }
    std::cout << std::endl;
}

static void print_cost(std::vector<size_t> &splits, interval_sum<double> &is, std::string prefix) {
    size_t start = 0;
    double cost = 0.0;
    for (size_t i = 0; i < splits.size(); ++i) {
        size_t end = splits[i];
        cost += is.cost_interval_l2(start, end);
        start = end + 1;
        if (i < splits.size() - 1)
            start = std::min(start, splits[i+1]);
    }
    std::cout << "[" << prefix << "] cost = " << cost << std::endl;
}

kmeans_lloyd::kmeans_lloyd() : mt(std::time(nullptr)) { }
std::string kmeans_lloyd::name() { return std::string("lloyd"); }

void kmeans_lloyd::set_seed(std::mt19937::result_type val) {
    mt.seed(val);
}

std::mt19937::result_type kmeans_lloyd::random_value() {
    auto val = mt();
    return val;
}

std::vector<size_t> kmeans_lloyd::init_splits(size_t n, size_t k) {
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

kmeans_lloyd_slow::kmeans_lloyd_slow(const std::vector<double> &points) : points(points), is(points), n(points.size()) {}

std::unique_ptr<kmeans_result> kmeans_lloyd_slow::compute(size_t k) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);

    if (k == 1) {res->cost = is.cost_interval_l2(0, n-1); return res;}

    std::vector<size_t> splits = this->init_splits(n, k); // splits[i] is the index of the
                                                          // last point in cluster i.

    bool converged = false;
    while (!converged) {
        std::vector<double> new_means;
        size_t start = 0;
        for (size_t end : splits) {
            double mean = ((double) is.query(start, end + 1)) / (end + 1 - start);
            new_means.push_back(mean);
            start = end + 1;
            start = std::min(start, n-1);
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
        cost += is.cost_interval_l2(start, splits[i]);
        res->centers.push_back(is.query(start, splits[i]) / (splits[i] - start + 1));
        start = splits[i]+1;
        if (i < k - 1)
            start = std::min(start, splits[i+1]);
    }
    res->cost = cost;
    return res;
}

std::unique_ptr<kmeans_result> kmeans_lloyd_slow::compute_and_report(size_t k) {
    return compute(k);
}

kmeans_lloyd_fast::kmeans_lloyd_fast(const std::vector<double> &points) : points(points),
                                                                          is(points),
                                                                          n(points.size()) {}

std::unique_ptr<kmeans_result> kmeans_lloyd_fast::compute(size_t k) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);

    if (k == 1) {
        res->cost = is.cost_interval_l2(0, n-1);
        return res;
    }
    std::vector<size_t> splits = this->init_splits(n, k); // splits[i] is the index of the
                                                          // last point in cluster i.
    bool converged = false;
    while (!converged) {
        size_t first_point = 0;
        bool change_detected = false;
        std::vector<double> means(k, 0.0);
        {
            size_t first_point = 0;
            for (size_t i = 0; i < k; ++i) {
                size_t last_point = splits[i];
                means[i] = is.query(first_point, last_point + 1) / (last_point - first_point + 1);
                first_point = last_point + 1;
                first_point = std::min(last_point + 1, n-1);
            }
        }
        for (size_t i = 0; i < k-1; ++i) {
            // binary search to update splits[i].
            double mean_curr = means[i];
            double mean_next = means[i+1];
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
        }
        for (size_t i = 1; i < k; ++i) {
            if (splits[i] == splits[i-1]) splits[i-1] = n-1;
        }
        std::stable_partition(splits.begin(), splits.end(), [&](size_t l){return l < n-1;});
        if (!change_detected) converged = true;

    }
    size_t start = 0;
    double cost = 0;
    for (size_t i = 0; i < k; ++i) {
        cost += is.cost_interval_l2(start, splits[i]);
        res->centers.push_back(is.query(start, splits[i]) / (splits[i] - start + 1));
        start = splits[i]+1;
        if (i < k-1)
            start = std::min(start, splits[i+1]);
    }
    res->cost = cost;
    return res;
}

std::unique_ptr<kmeans_result> kmeans_lloyd_fast::compute_and_report(size_t k) {
    return compute(k);
}

