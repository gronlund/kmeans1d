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

        std::tie(val_found, k_found) = this->with_smawk(n);
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

kmeans_hirschberg_larmore::~kmeans_hirschberg_larmore() {}

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

std::vector<size_t> kmeans_hirschberg_larmore::smawk_inner(std::vector<size_t> &columns, size_t e, std::vector<size_t> &rows) {
    // base case.
    size_t n = columns.size();
    size_t result_size = (n + e - 1) / e;
    if (rows.size() == 1) {
        // result is of length (n + e - 1)/e
        return std::vector<size_t>(result_size, 0);
    }

    //reduce
    std::vector<size_t> new_rows;
    std::vector<size_t> translate;
    if (result_size < rows.size()) {
        for (size_t i = 0; i < rows.size(); ++i) {
            // I1: forall j in [0..new_rows.size() - 2]: g(new_rows[j], columns[e*j]) < g(new_rows[j+1], columns[e*j]).
            // for (size_t j = 1; j < new_rows.size(); ++j) {
            //     assert(g(new_rows[j-1], columns[e*(j-1)]) < g(new_rows[j], columns[e*(j-1)]));
            // }
            // I2: every column minima is either already in a row in new_rows OR
            //                         it is in rows[j] where j >= i.
            auto r = rows[i];
            //&& (new_rows.size() - 1 + (rows.size() - i)) >= result_size
            while (new_rows.size() &&
                   g(r, columns[e * (new_rows.size() - 1)]) <= g(new_rows.back(), columns[e * (new_rows.size() - 1)])) {
                new_rows.pop_back();
                translate.pop_back();
            }
            if (e * new_rows.size() < n) { new_rows.push_back(r); translate.push_back(i); }
        }
    } else {
        new_rows = rows;
        for (size_t i = 0; i < rows.size(); ++i) translate.push_back(i);
    }
    // assert(new_rows.size() <= result_size); // new_row.size() = ceil(n/e)
    // assert(new_rows.size());
    if (result_size == 1) {
        return std::vector<size_t>{translate[0]};
    }
    //recurse
    std::vector<size_t> column_minima_rec = smawk_inner(columns, 2*e, new_rows);  // indexes in new_rows
    // assert(column_minima_rec.size() == ((result_size + 1)/ 2));
    std::vector<size_t> column_minima; // indexes in rows

    //combine.
    column_minima.push_back(translate[column_minima_rec[0]]);
    for (size_t i = 1; i < column_minima_rec.size(); ++i) {
        size_t from = column_minima_rec[i-1];  // index in new_rows
        size_t to = column_minima_rec[i]; // index in new_rows
        size_t new_column = (2 * i - 1); // 1, 3, 5..

        // assert(column_minima.size() == new_column);

        column_minima.push_back(from);
        for (size_t r = from; r <= to; ++r) {
            if (g(new_rows[r], columns[new_column*e]) <= g(rows[column_minima[new_column]], columns[new_column*e])) {
                column_minima[new_column] = translate[r];
            }
        }
        column_minima.push_back(translate[to]);
    }
    // assert(column_minima.size() == result_size || column_minima.size() == result_size - 1);
    if (column_minima.size() < result_size) {
        // assert(column_minima.size() == result_size - 1);
        size_t from = column_minima_rec.back();
        size_t new_column = column_minima.size();

        column_minima.push_back(translate[from]);
        for (size_t r = from; r < new_rows.size(); ++r) {
            if (g(new_rows[r], columns[new_column*e]) <= g(rows[column_minima[new_column]], columns[new_column*e])) {
                column_minima[new_column] = translate[r];
            }
        }
    }
    // assert(column_minima.size() == result_size);
    return column_minima;
}

std::vector<double> kmeans_hirschberg_larmore::smawk(size_t i0, size_t i1, size_t j0, size_t j1, std::vector<size_t> &idxes) {
    std::vector<size_t> rows, cols;
    for (size_t i = i0; i <= i1; ++i) {
        rows.push_back(i);
    }
    for (size_t j = j0; j <= j1; ++j) {
        cols.push_back(j);
    }
    std::vector<size_t> column_minima = smawk_inner(cols, 1, rows); // indexes in rows.
    std::vector<double> res(column_minima.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = g(rows[column_minima[i]], cols[i]);
        idxes.push_back(rows[column_minima[i]]);
        assert(res[i] != std::numeric_limits<double>::max());
    }
    return res;
}
/**
 * @return f[j] = smallest value in column j0 <= j <= j1 of submatrix G[i0:i1,j0:j1] all inclusive.
 */
std::vector<double> kmeans_hirschberg_larmore::smawk_naive(size_t i0, size_t i1, size_t j0, size_t j1, std::vector<size_t> &idxes) {
    std::vector<double> column_minima(j1 - j0 + 1, std::numeric_limits<double>::max());
    idxes.resize(j1 - j0 + 1, n+10);
    for (size_t j = j0; j<= j1; ++j) {
        for (size_t i = i0; i <= i1; ++i) {
            if (i >= j) continue; // g(i, j) is infinity.
            double val = g(i, j);
            if (val < column_minima[j-j0]) {
                column_minima[j-j0] = val;
                idxes[j-j0] = i;
            }
        }
    }
    return column_minima;
}

std::pair<double, size_t> kmeans_hirschberg_larmore::with_smawk(size_t n) {
    std::cout << "call with_smawk lambda=" << lambda << std::endl;
    f.resize(n+1, 0);
    bestleft.resize(n+1, 0);
    f[0] = 0;
    size_t c = 0, r = 0;

    while (c < n) {
        //std::cout << "hello" << std::endl;
        // step 1
        size_t p = std::min(2*c - r + 1, n);
        // step 2
        {
            std::vector<size_t> bl;
            std::vector<double> column_minima = smawk(r, c, c+1, p, bl);
            for (size_t j = c+1; j <= p; ++j) {
                f[j] = column_minima[j-(c+1)];
                bestleft[j] = bl[j-(c+1)];
            }
        }
        // step 3
        if (c+1 <= p-1) {
            std::vector<size_t> bl;
            std::vector<double> H = smawk(c+1, p-1, c+2, p, bl);
            // step 4
            size_t j0 = p+1;
            for (size_t j = p; j >= c+2; --j) {
                if (H[j-(c+2)] < f[j]) j0 = j;
            }
            //step 5
            if (j0 == p+1) {
                c = p;
            } else {
                f[j0] = H[j0-(c+2)];
                bestleft[j0] = bl[j0-(c+2)];
                r = c + 1;
                c = j0;
            }
        } else {
            c = p;
        }
    }
    // find length
    size_t m = n;
    size_t length = 0;
    while (m > 0) {
        m = bestleft[m];
        ++length;
    }
    return std::make_pair(f[n], length);

}

