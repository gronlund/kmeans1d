#include "stdint.h"
#include "assert.h"
#include <cstdlib>
#include <vector>
#include <limits>

#include "main.h"

using std::free; using std::calloc;
using std::vector;

static double oo = std::numeric_limits<double>::infinity();

Table::Table(size_t k, size_t n, double init_val) : k(k), n(n), arr(n*k, 0) {
    for (size_t i = 0; i < n*k; ++i) {
        arr[i] = init_val;
    }
}

double Table::get(size_t i, size_t j) {
    size_t idx = i*n+j;
    return arr[i*n+j];
}


void Table::set(size_t i, size_t j, double val) {
    size_t idx = i*n+j;
    assert(idx < k*n);
    arr[i*n + j] = val;
}



IntervalSum::IntervalSum(const double *points, size_t n) {
    prefix_sum = (double*) calloc(n+1, sizeof(double));
    prefix_sum_of_squares = (double*) calloc(n+1, sizeof(double));
    double prefix_sum_val = 0;
    double prefix_square_val = 0;
    for (size_t i = 1; i <= n; ++i) {
        prefix_sum_val += points[i-1];
        prefix_square_val += points[i-1]*points[i-1];
        prefix_sum[i] = prefix_sum_val;
        prefix_sum_of_squares[i] = prefix_square_val;
    }
}

double IntervalSum::query_sq(size_t i, size_t j) {
    return prefix_sum_of_squares[j]-prefix_sum_of_squares[i];
}

/**
 * @returns sum from i to j of points[l] where i included, j not included.
 */
double IntervalSum::query(size_t i, size_t j) {
    return prefix_sum[j]-prefix_sum[i];
}

IntervalSum::~IntervalSum() {
    free(prefix_sum);
    free(prefix_sum_of_squares);
}


double kmeans1d_cost_interval_l1(IntervalSum &IS, int64_t s, size_t i) {
    size_t n = i-s+1;
    size_t mid = s+n/2;
    double prefix_cost = IS.query(s, mid);
    double suffix_cost = IS.query(mid+1, i+1);
    double cost = suffix_cost - prefix_cost;
    if ((n % 2) == 0) {
        double mid_cost = IS.query(mid, mid+1);
        cost += mid_cost;
    }
    return cost;
}

double kmeans1d_cost_interval_l2(IntervalSum &IS, int64_t s, size_t i) {
    double suffix_sum = IS.query(s, i+1);
    double suffix_sq = IS.query_sq(s, i+1);
    double mean = suffix_sum / (i-s+1);
    size_t length = i-s+1;
    double interval_cost = suffix_sq + mean*mean*length - 2 * suffix_sum * mean;
    return interval_cost;
}

void kmeans_init_table(double *points, size_t n, size_t k,
		       Table &dp_table, IntervalSum &IS, int norm) {

    for (size_t i = 0; i < n; ++i) {
        switch (norm) {
        case 1: {
            double cost = kmeans1d_cost_interval_l1(IS, 0, i);
            dp_table.set(1, i, cost);
            break;
        }
        case 2: {
            double cost = kmeans1d_cost_interval_l2(IS, 0, i);
            dp_table.set(1, i, cost);
            break;
        }
        }
    }

    dp_table.set(1, 0, 0.0);

    for (size_t j = 1; j <= k; ++j) {
        dp_table.set(j, 0, 0.0);
    }
}

/**
 * @returns cost of clustering with c clusters, when last cluster is points[s..i].
 */
double kmeans1d_cost(Table &dp_table, IntervalSum &IS,
		     int64_t s, size_t i, size_t c, int norm) {
    double interval_cost;
    switch (norm) {
    case 1:
        interval_cost = kmeans1d_cost_interval_l1(IS, s, i);
        break;
    case 2:
        interval_cost = kmeans1d_cost_interval_l2(IS, s, i);
        break;
    }
    if (s == 0) {
        if (c == 0) return oo;
        return interval_cost;
    } else {
        if (dp_table.get(c-1, s-1) == oo) {
            return oo;
        }
        return interval_cost + dp_table.get(c-1, s-1);
    }
}

void kmeans1d_solve_for_k(double *points, size_t n, Table &dp_table, IntervalSum &IS,
                          size_t begin, size_t end, size_t c,
			  int64_t split_left, int64_t split_right, int norm) {

    size_t mid = (begin+end)/2;

    double best = oo;
    int64_t best_split = mid;
    for (int64_t s = split_left; s <= split_right && s <= mid; ++s) {
        double cost = kmeans1d_cost(dp_table, IS, s, mid, c, norm);
        if (cost < best) {
            best = cost;
            best_split = s;
        }
    }

    dp_table.set(c, mid, best);

    if (mid != begin) {
        kmeans1d_solve_for_k(points, n, dp_table, IS,
                             begin, mid, c,
                             split_left, best_split, norm);
    }
    if (mid + 1 != end) {
        kmeans1d_solve_for_k(points, n, dp_table, IS,
                             mid+1, end, c,
                             best_split, split_right, norm);
    }
}

void kmeans1d_solve(double *points, size_t n, size_t k,
		    Table &dp_table, IntervalSum &IS, int norm) {

    kmeans_init_table(points, n, k, dp_table, IS, norm);
    for (size_t c = 2; c <= k; ++c) {
        kmeans1d_solve_for_k(points, n, dp_table, IS,
                             0, n, c,
                             0, n-1, norm);
    }
}

void kmeas1d_solve_faster(double *points, size_t n, size_t k,
                          Table &dp_table, IntervalSum &IS, int norm) {
    return;
}

void kmeans_cluster_centers(double *points, size_t n, size_t k, Table &dp_table,
			    IntervalSum &IS, double *clusters, int norm) {
    size_t end = n; // invariant: end is one past end of cluster.
    double search_val = dp_table.get(k, n-1);
    for (size_t i = k; i > 0; --i) {
        for (size_t j = 0; j < end; ++j) {
            size_t idx = end - j - 1; // index of first point in cluster.
            double cost = kmeans1d_cost(dp_table, IS, idx, end-1, i, norm);
            if (cost == search_val) {
                switch (norm) {
                case 1:
                    clusters[i-1] = points[(end-idx)/2 + idx];
                    break;
                case 2:
                    clusters[i-1] = IS.query(idx, end) / (end - idx);
                    break;
                }
                end = idx;
                if (idx == 0) {
                    search_val = 0;
                } else {
                    search_val = dp_table.get(i-1, idx-1);
                }
                break;
            }
        }
    }
}

/**
 * Assigns each point to a cluster.
 * cluster_membership is of length n, and cluster_membership[i]
 * is the cluster point i has been assigned to.
 * Every point is assigned to the nearest cluster center.
 * @parameter cluster_centers is a ascending array with the cluster centers.
 */
void kmeans_cluster_membership(double *points, double *cluster_centers,
                               int *cluster_membership, size_t n, size_t k,
                               int *permutation) {

    size_t current = 0;
    size_t i = 0;

    while (i < n) {
        if (current == k-1) break;
        double dist_cur = points[i]-cluster_centers[current];
        dist_cur = dist_cur*dist_cur;
        double dist_next = points[i]-cluster_centers[current+1];
        dist_next = dist_next*dist_next;
        if (dist_cur < dist_next) {
            cluster_membership[permutation[i]] = current;
        } else {
            ++current;
            cluster_membership[permutation[i]] = current;
        }
        ++i;
    }

    while (i < n) {
        cluster_membership[permutation[i]] = k-1;
        ++i;
    }
}
/*
extern "C" {

    // assume k > 0
    // assume n > 1
    void kmeans1d(double *points, int *n, int *k,
                  double *optimum_values, double *cluster_centers,
                  int *cluster_membership, int *norm) {

        int *permutation = (int*) calloc(n, sizeof(int));
        for (size_t i = 0; i < *n; ++i) {
            permutation[i] = i;
        }

        rsort_with_index(points, permutation, *n);
        IntervalSum IS(points, *n);
        Table dp_table(*k + 1, *n, oo);
        kmeans1d_solve(points, *n, *k, dp_table, IS, *norm);
        for (size_t i = 1; i <= *k; ++i) {
            optimum_values[i-1] = dp_table.get(i, *n - 1);
        }
        size_t used = 0;
        for (size_t i = 1; i <= *k; ++i) {
            kmeans_cluster_centers(points, *n, i, dp_table, IS, cluster_centers + used, *norm);
            kmeans_cluster_membership(points, cluster_centers + used,
                                      cluster_membership + (i-1)*(*n),
                                      *n, i, permutation);
            used += i;
        }
        free(permutation);
    }
}

*/
