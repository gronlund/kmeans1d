#include <algorithm>
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
#include "interval_sum.hpp"

static double lambda = 0;
static interval_sum<double> is;

static std::vector<double> f;
static std::vector<size_t> bestleft;

kmeans_hirschberg_larmore::kmeans_hirschberg_larmore(const std::vector<double> &points) :
    f(points.size() + 1, 0.0), bestleft(points.size() + 1, 0),
    is(points), points(points), n(points.size()) { }

std::string kmeans_hirschberg_larmore::name() { return std::string("hirc"); }

std::unique_ptr<kmeans_result> kmeans_hirschberg_larmore::compute(size_t k) {
    std::unique_ptr<kmeans_result> kmeans_res(new kmeans_result);
    if (k >= n) {
        kmeans_res->cost = 0.0;
        kmeans_res->centers.resize(k);
        for (size_t i = 0; i < n; ++i) {
            kmeans_res->centers[i] = points[i];
        }
        for (size_t i = n; i < k; ++i) {
            kmeans_res->centers[i] = points[n-1];
        }
        return kmeans_res;
    }
    if (k == 1) {
        kmeans_res->cost = is.cost_interval_l2(0, n-1);
        kmeans_res->centers.push_back(is.query(0, n) / ((double) n));
        return kmeans_res;
    }

    double lo = 0.0;
    double hi = is.cost_interval_l2(0, n-1);
    double hi_intercept = hi;
    double lo_intercept = 0;
    size_t hi_k = 1;
    size_t lo_k = n;
    //double hi = 1e-2;

    double val_found, val_found2;
    size_t k_found, k_found2;
    size_t cnt = 0;
    while (true) {
        ++cnt;
        double t = (hi_intercept - lo_intercept) / sqrt(lo_k - hi_k);
        double intercept_guess = (hi_intercept + lo_intercept) / 2;
        double intersect_hi = (intercept_guess - hi_intercept) / (hi_k - k);
        double intersect_lo = (intercept_guess - lo_intercept) / (lo_k - k);
        assert(intercept_guess > 0);
        assert(intercept_guess <= hi_intercept);
        assert(intercept_guess >= lo_intercept);
        lambda = (hi_intercept - lo_intercept) / (lo_k - hi_k);

        std::tie(val_found, k_found) = this->basic(n);
        if (k_found > k) {
            lo_k = k_found;
            lo = lambda;
            lo_intercept = val_found - lo_k * lambda;
        } else if (k_found < k) {
            hi = lambda;
            hi_k = k_found;
            hi_intercept = val_found - hi_k * lambda;
        } else {
            hi = lambda;
            break;
        }
    }
    assert(k == k_found);
    get_actual_cost(n, kmeans_res);
    return kmeans_res;
}

std::unique_ptr<kmeans_result> kmeans_hirschberg_larmore::compute_and_report(size_t k) {
    return compute(k);
}

double kmeans_hirschberg_larmore::weight(size_t i, size_t j) {
    if (i >= j) return std::numeric_limits<double>::max();
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
        double gim = g(i, mid);
        double gjm = g(j, mid);
        double gkm = g(k, mid);
        if (gim <= gjm) {
            lo = mid;
            if (gkm <= gjm) return true;
        } else {
            hi = mid;
            if (gjm < gkm) return false;
        }
    }
    bool result = (g(k, hi) <= g(j, hi));
    return result;
}

std::pair<double, size_t> kmeans_hirschberg_larmore::basic(size_t n) {
    std::cout << "call basic lambda=" << lambda << std::endl;
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

double kmeans_hirschberg_larmore::get_actual_cost(size_t n, std::unique_ptr<kmeans_result> &res) {
    double cost = 0.0;
    size_t m = n;

    std::vector<double> centers;
    while (m != 0) {
        size_t prev = bestleft[m];
        cost += is.cost_interval_l2(prev, m-1);
        double avg = is.query(prev, m) / (m - prev);
        centers.push_back(avg);
        m = prev;
    }

    res->centers.resize(centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
        res->centers[i] = centers[centers.size() - i - 1];
    }
    res->cost = cost;
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


