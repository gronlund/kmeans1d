#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "kmeans.h"
#include "common.h"
#include "interval_sum.hpp"

static double lambda = 0;
static interval_sum<double> is;

static std::vector<double> f;
static std::vector<size_t> bestleft;

kmeans_hirschberg_larmore::kmeans_hirschberg_larmore(const std::vector<double> &points) :
    f(points.size() + 1, 0.0), bestleft(points.size() + 1, 0),
    is(points), n(points.size()) { }

std::unique_ptr<kmeans_result> kmeans_hirschberg_larmore::compute(size_t k) {
    std::unique_ptr<kmeans_result> kmeans_res(new kmeans_result);
    if (k == 1) {
        kmeans_res->cost = is.cost_interval_l2(0, n-1);
        kmeans_res->centers.push_back(is.query(0, n) / ((double) n));
        return kmeans_res;
    }

    double lo = 0.0;
    double hi = 2 * is.cost_interval_l2(0, n-1);

    double val_found, best_val;
    size_t k_found;
    while (hi - lo > 1e-10) {
        double mid = lo + (hi-lo) / 2.0;
        lambda = mid;
        std::tie(val_found, k_found) = this->basic(n);
        if (k_found > k) {
            lo = mid;
        } else if (k_found < k) {
            hi = mid;
        } else {
            hi = mid;
            break;
        }
    }
    lambda = hi;
    std::tie(val_found, k_found) = this->basic(n);
    assert(k == k_found);
    //double cost = val_found - k_found * lambda; // this is imprecise
    double cost = get_actual_cost(n, 0);
    kmeans_res->cost = cost;
    return kmeans_res;
}

kmeans_hirschberg_larmore::~kmeans_hirschberg_larmore() {}

double kmeans_hirschberg_larmore::weight(size_t i, size_t j) {
    assert(i < j);
    return is.cost_interval_l2(i, j-1) + lambda;
}

double kmeans_hirschberg_larmore::g(size_t i, size_t j) {
    return f[i] + weight(i, j);
}

bool kmeans_hirschberg_larmore::bridge(size_t i, size_t j, size_t k, size_t n) {
    if (k == n) {
        return true;
    }
    if (g(i, n) <= g(j, n)) {
        return true;
    }
    size_t lo = k;
    size_t hi = n;
    while (hi - lo >= 2) {
        size_t mid = lo + (hi-lo)/2;
        if (g(i, mid) <= g(j, mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bool result = (g(k, hi) <= g(j, hi));
    return result;
}

std::pair<double, size_t> kmeans_hirschberg_larmore::basic(size_t n) {
    f.resize(n+1, 0);
    for (size_t i = 0; i <= n; ++i) f[i] = 0;
    bestleft.resize(n+1, 0);
    for (size_t i = 0; i <= n; ++i) bestleft[i] = 0;
    std::vector<size_t> D = {0};
    size_t front = 0;
    for (size_t m = 1; m <= n-1; ++m) {
        f[m] = g(D[front], m);
        bestleft[m] = D[front];
        while (front + 1 < D.size() && g(D[front + 1], m+1) <= g(D[front], m+1)) {
            ++front;
        }
        if (g(m, n) < g(D[D.size() - 1], n)) {
            D.push_back(m);
        } else { continue; }
        while (front + 2 < D.size() && bridge(D[D.size() - 3], D[D.size() - 2], m, n))  {
            std::swap(D[D.size() - 1], D[D.size() - 2]);

            D.pop_back();
        }
        if (front + 2 == D.size() && g(D[D.size() - 1], m+1) <= g(D[D.size() - 2], m+1)) {
            ++front;
        }
    }
    assert(front + 1 == D.size());
    f[n] = g(D[front], n);
    bestleft[n] = D[front];

    // find length.
    size_t m = n;
    size_t length = 0;
    while (m > 0) {
        m = bestleft[m];
        ++length;
    }
    return std::make_pair(f[n], length);
}

double kmeans_hirschberg_larmore::get_actual_cost(size_t n, double *centers_ptr) {
    double cost = 0.0;
    size_t m = n;

    std::vector<double> centers;
    while (m != 0) {
        size_t prev = bestleft[m];
        cost += is.cost_interval_l2(prev, m-1);
        double avg = is.query(prev, m-1) / (m - prev - 1);
        centers.push_back(avg);
        m = prev;

    }
    if (centers_ptr) {
        for (size_t i = 0; i < centers.size(); ++i) {
            centers_ptr[i] = centers[centers.size() - i - 1];
        }
    }
    return cost;
}

std::pair<double, size_t> kmeans_hirschberg_larmore::traditional(size_t n) {
    f.resize(n, 0);
    for (size_t i = 0; i < n; ++i) f[i] = 0;
    bestleft.resize(n, 0);
    for (size_t i = 0; i < n; ++i) bestleft[i] = 0;

    for (size_t m = 1; m < n; ++m) {
        f[m] = g(0, m);
        bestleft[m] = 0;
        for (size_t i = 1; i < m; ++i) {
            if (g(i, m) < f[m]) {
                f[m] = g(i, m);
                bestleft[m] = i;
            }
        }
    }
    size_t m = n-1;
    size_t length = 0;
    while (m > 0) {
        m = bestleft[m];
        ++length;
    }
    return std::make_pair(f[n-1], length);
}

static double kmeans_object_oriented(double *points_ptr, size_t n, double *centers_ptr, size_t k) {
    std::vector<double> points(n, 0);
    for (size_t i = 0; i < n; ++i) points[i] = points_ptr[i];
    kmeans_hirschberg_larmore kmeans_hirsch(points);
    std::unique_ptr<kmeans_result> kmeans_res = kmeans_hirsch.compute(k);
    if (centers_ptr) {
        for (size_t i = 0; i < k; ++i) centers_ptr[i] = kmeans_res->centers[i];
    }
    return kmeans_res->cost;
}


kmeans_fn get_kmeans_hirsch_larmore() {
    return kmeans_object_oriented;
}
