#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "kmeans.h"
#include "common.h"

static double lambda = 0;
static IntervalSum is;

static std::vector<double> f;
static std::vector<size_t> bestleft;



static void init(double *points_input, size_t n) {
    init_IntervalSum(&is, points_input, n);
}

static double weight(size_t i, size_t j) {
    assert(i < j);
    return cost_interval_l2(&is, i, j-1) + lambda;
}

static double g(size_t i, size_t j) {
    return f[i] + weight(i, j);
}

static bool bridge(size_t i, size_t j, size_t k, size_t n) {
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

static std::pair<double, size_t> traditional(size_t n) {
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

static std::pair<double, size_t> basic(size_t n) {
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

static double get_actual_cost(size_t n, double *centers_ptr) {
    double cost = 0.0;
    size_t m = n;

    std::vector<double> centers;
    while (m != 0) {
        size_t prev = bestleft[m];
        cost += cost_interval_l2(&is, prev, m-1);
        double avg = query(&is, prev, m-1) / (m - prev - 1);
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

static double kmeans(double *points, size_t n, double *centers_ptr, size_t k) {
    init(points, n);
    if (k == 1) { return cost_interval_l2(&is, 0, n-1); }

    double lo = 0.0;
    double hi = 2 * cost_interval_l2(&is, 0, n-1);

    double val_found, best_val;
    size_t k_found;
    while (hi - lo > 1e-10) {
        double mid = lo + (hi-lo) / 2.0;
        lambda = mid;
        std::tie(val_found, k_found) = basic(n);
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
    std::tie(val_found, k_found) = basic(n);
    assert(k == k_found);
    //double cost = val_found - k_found * lambda; // this is imprecise
    double cost = get_actual_cost(n, centers_ptr);
    free_IntervalSum(&is);
    return cost;
}


kmeans_fn get_kmeans_hirsch_larmore() {
    return kmeans;
}
