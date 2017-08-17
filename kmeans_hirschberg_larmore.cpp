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

using namespace std;


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
    if (k == n-1) {
        return true;
    }
    if (g(i, n-1) <= g(j, n-1)) {
        return true;
    }
    size_t lo = k;
    size_t hi = n-1;
    while (hi - lo >= 2) {
        size_t mid = lo + (hi-lo)/2;
        if (g(i, mid) <= g(j, mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bool result = (g(k, hi) <= g(j, hi));
#ifdef DEBUG
    cout << "Bridge(" << i << ", " << j
         << ", " << k << ", " << n
         << ") = " << (result?"True":"False")
         << "    hi: " << hi << endl;
#endif
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
    f.resize(n, 0);
    for (size_t i = 0; i <= n; ++i) f[i] = 0;
    bestleft.resize(n, 0);
    for (size_t i = 0; i <= n; ++i) bestleft[i] = 0;
    std::vector<size_t> D = {0};
    size_t front = 0;
    for (size_t m = 1; m <= n-1; ++m) {
        f[m] = g(D[front], m);
        bestleft[m] = D[front];
        while (front + 1 < D.size() && g(D[front + 1], m+1) <= g(D[front], m+1)) {
            ++front;
        }
        while (front + 1 < D.size() && bridge(D[D.size() - 2], D[D.size() - 1], m, n + 1))  {
            D.pop_back();
        }
        if (g(m, n) < g(D[D.size() - 1], n)) {
            D.push_back(m);
        }
    }
    assert(front + 1 == D.size());
    f[n] = g(D[front], n);
    bestleft[n] = D[front];
#ifdef DEBUG
    cout << "f        = [";
    for (size_t i = 0; i < n; ++i) {
        cout << setprecision(3) << f[i] << ", ";
    }
    cout << "]" << endl;
    cout << "bestleft = [";
    for (size_t i = 0; i < n; ++i) {
        cout << bestleft[i] << ", ";
    }
    cout << "]" << endl;
    cout << "indices  = [";
    for (size_t i = 0; i < n; ++i) {
        cout << i << ", ";
    }
    cout << "]" << endl;

    cout << "find length" << endl;
#endif

    // find length.
    size_t m = n-1;
    size_t length = 0;
    while (m > 0) {
        m = bestleft[m];
        ++length;
    }
#ifdef DEBUG
    cout << "returning" << endl;
#endif
    return std::make_pair(f[n-1], length);
}

static double get_actual_cost(size_t n) {
    double cost = 0.0;
    size_t m = n-1;

    std::vector<double> centers;
    while (m != 0) {
        size_t prev = bestleft[m];
        cost += cost_interval_l2(&is, prev, m-1);
        double avg = query(&is, prev, m-1) / (m - prev - 1);
        centers.push_back(avg);
        m = prev;
    }
#ifdef DEBUG
    cout << "centers = [";
    for (size_t i = 0; i < centers.size(); ++i) {
        cout << centers[centers.size() - i - 1] << ", ";
    }
    cout << "]" << endl;
#endif
    return cost;
}

static double kmeans(double *points, size_t n, double *last_row, size_t k) {
    init(points, n);
#ifdef DEBUG
    cout << "--------------- K = " << k << " -------------------------------" << endl;
#endif
    if (k == 1) { return cost_interval_l2(&is, 0, n-1); }
    //if (k != 2) return 0.0;
    double lo = 0.0;
    double hi = std::numeric_limits<double>::max();
    hi = 10;
    double val_found, best_val;
    size_t k_found;
    while (hi - lo > 1e-10) {
        double mid = lo + (hi-lo) / 2.0;
        lambda = mid;
        lambda = 0;
#ifdef DEBUG
        std::cout << std::setprecision(5) << "lo = " << lo << "   hi = " << hi << std::endl;
        std::cout << "lambda = " << lambda << endl;
#endif
        std::tie(val_found, k_found) = basic(n);
        //std::tie(val_found, k_found) = traditional(n + 1);
        double cost = get_actual_cost(n + 1);
#ifdef DEBUG
        cout << "actual cost: " << cost << "   k_found = " << k_found << endl;
#endif
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
    //std::tie(val_found, k_found) = traditional(n + 1);
    double cost = get_actual_cost(n + 1);
#ifdef DEBUG
    if (cost != val_found - k * lambda) {
        cout << "val_found does not equal actual minus k * lambda" << endl;
        cout << val_found << " - " << k*lambda << " - "
             << cost << " = " << val_found - k * lambda - cost << endl;
    }

    cout << "actual cost: " << cost << "   k_found = " << k_found << endl;
#endif
    //assert(k == k_found);
    free_IntervalSum(&is);
    return cost;
}


kmeans_fn get_kmeans_hirsch_larmore() {
    return kmeans;
}
