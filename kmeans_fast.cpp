#include "kmeans.h"
#include <cassert>
#include <limits>
/**
 * This code implements the matrix searching algorithm
 * for computing k-means.
 */
static double oo = std::numeric_limits<double>::max();

kmeans_fast::kmeans_fast(const std::vector<double> &points) : kmeans_dp(points) { }

std::string kmeans_fast::name() { return std::string("fast"); }
std::unique_ptr<kmeans_dp> kmeans_fast::get_instance(std::vector<double> &points) {
    return std::unique_ptr<kmeans_dp>(new kmeans_fast(points));
}

std::unique_ptr<kmeans_result> kmeans_fast::compute(size_t k) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);
    res->cost = 0.0;

    base_case(k);
    for (size_t c = 2; c <= k; ++c) {
        std::swap(row_prev, row);
        fill_row(c);
    }
    res->cost = row[n-1];
    return res;
}

/**
 * @param i is the number of clusters / row of the matrix.
 * @param m is the last point of the clustering.
 * @param j is the first point of the last cluster.
 * @return C_i[m][j] in the the 1D kmeans paper.
 */
double kmeans_fast::cimj(size_t i, size_t m, size_t j) {
    assert(i > 0);
    if (m < j) {
        return row_prev[m];
    } else {
        if (j == 0) {
            return is.cost_interval_l2(0, m);
        }
        double best_before = row_prev[j-1];
        double last_cluster_cost = is.cost_interval_l2(j, m);
        return last_cluster_cost + best_before;
    }
}

void kmeans_fast::reduce(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
                         std::vector<size_t> &cols_output, size_t reduce_i) {
    // n rows, m columns.
    // output is n rows and n columns.
#ifdef DEBUG_KMEANS
    printf("[reduce] called with n=%d, m=%d, reduce_i=%d\n", n, m, reduce_i);
#endif
    std::vector<size_t> prev_col(m, 0);
    std::vector<size_t> next_col(m, 0);
    for (size_t i = 1; i < m; ++i) {
        prev_col[i] = i-1;
        next_col[i] = i+1;
    }
    next_col[0] = 1;
    prev_col[0] = m;

    size_t remaining_columns = m;
    size_t rowk = 0; // index in rows
    size_t colk = 0; // index in cols.
    while (remaining_columns > n) {
        double val = -cimj(reduce_i, row_multiplier * rowk, cols[colk]);
        //printf("rowk=%ld\n", rowk);
        double next_val = -cimj(reduce_i, row_multiplier * rowk,
                                cols[next_col[colk]]);
        if (val >= next_val && rowk < n-1) {
            rowk += 1;
            colk = next_col[colk];
            assert(colk < m);
        } else if (val >= next_val && rowk == n-1) {
            // delete column next_col[colk].
            // i.e. update the pointers.
            size_t to_delete = next_col[colk];
            //assert(to_delete < m);
            size_t next = next_col[to_delete];
            size_t prev = prev_col[to_delete];
            assert(prev == colk);
            if (next != m) {
                prev_col[next] = prev;
            }
            if (prev != m) {
                next_col[prev] = next;
            }
            next_col[to_delete] = m;
            prev_col[to_delete] = m;
            --remaining_columns;
        } else if (val < next_val) {
            // First adjust pointers. Need to use old pointers later.
            size_t old_colk = colk;
            if (rowk > 0) {
                --rowk;
                colk = prev_col[colk];
            } else {
                colk = next_col[colk];
            }

            // delete column colk, which means update the pointers.
            size_t prev = prev_col[old_colk];
            size_t next = next_col[old_colk];
            if (prev != m) { // meaning the previous exists.
                assert(next_col[prev] == old_colk);
                next_col[prev] = next_col[old_colk];
            }
            if (next != m) { // meaning next exists.
                assert(prev_col[next] == old_colk);
                prev_col[next] = prev_col[old_colk];
            }
            prev_col[old_colk] = m;
            next_col[old_colk] = m;
            --remaining_columns;
        }
    }

    // generate output.
    size_t j = 0;
    for (size_t i = 0; i < m; ++i) {
        if (prev_col[i] != m || next_col[i] != m) {
            cols_output[j] = cols[i];
            ++j;
        }
    }
}

void kmeans_fast::mincompute(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
                             size_t reduce_i, std::vector<size_t> &cols_output) {
#ifdef DEBUG_KMEANS
    printf("[mincompute] Called with n=%d, m=%d, reduce_i=%d\n", n, m, reduce_i);
#endif
    if (n == 1) {
        size_t r = 0; // r = rows[0]
        size_t idx = 0;
        double best = cimj(reduce_i, r, cols[0]);
        for (size_t i = 1; i < m; ++i) {
            size_t c = cols[i];
            double val = cimj(reduce_i, r, c);
            if (val < best) {
                best = val;
                idx = i;
            }
        }
        cols_output[0] = cols[idx];
#ifdef DEBUG_KMEANS
        printf("[mincompute] returning\n", n, m, reduce_i);
#endif
        return;
    }

    std::vector<size_t> cols_output_reduce(n, 0);
    reduce(row_multiplier, cols, n, m, cols_output_reduce, reduce_i);
    size_t n_rec = (n + 1) / 2;
    std::vector<size_t> output(n_rec, 0);
    mincompute(row_multiplier * 2, cols_output_reduce, n_rec, n, reduce_i, output);
#ifdef DEBUG_KMEANS
    printf("[mincompute] n = %d\n", n);
    for (size_t i = 0; i < n_rec; ++i) {
        printf("[mincompute] output[%ld] = %ld\n", i, output[i]);
    }
#endif
    { std::vector<size_t> empty; std::swap(empty, cols_output_reduce); }

    size_t first = 0; // index into cols.
    while (output[0] != cols[first]) ++first;

    // iterate odd rows.
    for (size_t i = 1; i < n; i+=2) {
        size_t current = first; // index into cols
        size_t end = current; // index into cols
        if (i + 1 < n) {
            while (output[(i+1)/2] != cols[end]) ++end;
        } else {
            end = m-1;
        }

        size_t best_idx = current; // index into cols.
        double best = oo;
        for (size_t z = current; z <= end; ++z) {
            size_t rowsi = row_multiplier * i;
            double val = cimj(reduce_i, rowsi, cols[z]);
            if (val < best) {
                best = val;
                best_idx = z;
            }
        }
#ifdef DEBUG_KMEANS
        printf("[mincompute] %d: cols[best_idx] = %d\n", i, cols[best_idx]);
#endif
        cols_output[i] = cols[best_idx];
        first = end;
    }
    for (size_t i = 0; i < n; i+=2) cols_output[i] = output[i/2];
    { std::vector<size_t> empty; std::swap(empty, output); };
#ifdef DEBUG_KMEANS
    printf("[mincompute] reduce_i = %d   n = %d\n", reduce_i, n);
    for (size_t i = 0; i < n; ++i) {
        printf("[mincompute] cols_output[%d] = %d\n", i, cols_output[i]);
    }
    printf("[mincompute] returning\n");
#endif
    return;
}

void kmeans_fast::fill_row(size_t k) {
    std::vector<size_t> cols(n, 0);
    std::vector<size_t> output(n, 0);
    for (size_t i = 0; i < n; ++i) {
        cols[i] = i;
    }

#ifdef DEBUG_KMEANS
    printf("Matrix cimj:\n");
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%.2f  ", cimj(k, i, j));
        }
        printf("\n");
    }
    printf(" ------    fill_row2   k = %ld   ---------\n", k);
#endif
    mincompute(1, cols, n, n, k, output);

    for (size_t i = 0; i < n; ++i) {
#ifdef DEBUG_KMEANS
        printf("output[%d] = %d\n", i, output[i]);
#endif
        size_t row = i;
        size_t col = output[i];
        this->row[i] = cimj(k, row, col);
    }
}

void kmeans_fast::base_case(size_t k) {
    for (size_t i = 0; i < n; ++i) {
        double cost = is.cost_interval_l2(0, i);
        row[i] = cost;
    }
    row[0] = 0.0;
}

static double kmeans_object_oriented(double *points_ptr, size_t n,
                                     double *last_row, size_t k) {
    std::vector<double> points(n, 0);
    for (size_t i = 0; i < n; ++i) points[i] = points_ptr[i];
    kmeans_fast kmeans_algorithm(points);
    std::unique_ptr<kmeans_result> kmeans_res = kmeans_algorithm.compute(k);
    return kmeans_res->cost;
}

kmeans_fn get_kmeans_fast() {
    return kmeans_object_oriented;
}

