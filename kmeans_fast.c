#include "kmeans.h"
#include "float.h"
#include "common.h"
#include "assert.h"
#include "stdlib.h"

static double oo = DBL_MAX;

static struct Table t;
static struct IntervalSum ps;

static double CC(size_t j, size_t m) {
    return cost_interval_l2(&ps, j, m);
}

/**
 * @param is the number of clusters / row of the matrix.
 * @param m is the last point of the clustering.
 * @param j is the first point of the last cluster.
 * @return C_i[m][j]
 */
static double cimj(size_t i, size_t m, size_t j) {
    // approved for {1.0, 2.0, 3.0, 4.0, 5.0} and 2 clusters.
    assert(i > 0);
    if (m < j) {
        double best_before = get(&t, i-1, m);
        return best_before;
    } else {
        if (j == 0) {
            return CC(0, m);
        }
        double best_before = get(&t, i-1, j-1);
        double last_cluster_cost = CC(j, m);
        return last_cluster_cost + best_before;
    }

}


static void reduce(size_t *rows, size_t *cols, size_t n, size_t m,
                      size_t *cols_output, size_t reduce_i) {
    // n rows, m columns.
    // output is n rows and n columns.
#ifdef DEBUG
    printf("[reduce] n=%ld  m=%ld  reduce_i=%ld\n", n, m, reduce_i);
#endif
    size_t *prev_col = (size_t*) malloc(m * sizeof(size_t));
    size_t *next_col = (size_t*) malloc(m * sizeof(size_t));
    for (size_t i = 1; i < m; ++i) {
        prev_col[i] = i-1;
        next_col[i] = i+1;
    }
    next_col[0] = 1;
    prev_col[0] = m;

    size_t remaining_columns = m;
    size_t rowk = 0; // index in rows
    size_t colk = 0; // index in cols.
    //printf("[reduce] called with n=%d, m=%d, reduce_i=%d\n", n, m, reduce_i);
    while (remaining_columns > n) {
        double val = -cimj(reduce_i, rows[rowk], cols[colk]);
        //printf("rowk=%ld\n", rowk);
        double next_val = -cimj(reduce_i, rows[rowk], cols[next_col[colk]]);
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
    free(next_col);
    free(prev_col);
    return;
}

static void mincompute(size_t *rows, size_t *cols, size_t n, size_t m, size_t reduce_i,
                       size_t *cols_output) {
#ifdef DEBUG
    printf("[mincompute] Called with n=%d, m=%d, reduce_i=%d\n", n, m, reduce_i);
    printf("[mincompute] input rows: ");
    for (size_t i = 0; i < n; ++i) {
        printf("%d  ", rows[i]);
    }
    printf("\n");
#endif
    if (n == 1) {
        size_t r = rows[0];
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
#ifdef DEBUG
        printf("[mincompute] returning\n", n, m, reduce_i);
#endif
        return;
    }

    size_t *cols_output_reduce = (size_t *) malloc(n * sizeof(size_t));
    reduce(rows, cols, n, m, cols_output_reduce, reduce_i);
    size_t n_rec = (n + 1) / 2;
    size_t *rows_recurse = (size_t *) malloc(n_rec * sizeof(size_t));
    size_t *output = (size_t *) malloc(n_rec * sizeof(size_t));
    for (size_t i = 0; i < n_rec; i++) rows_recurse[i] = rows[2*i];
    mincompute(rows_recurse, cols_output_reduce, n_rec, n, reduce_i, output);
#ifdef DEBUG
    printf("[mincompute] n = %d\n", n);
    for (size_t i = 0; i < n_rec; ++i) {
        printf("[mincompute] output[%ld] = %ld\n", i, output[i]);
    }
#endif
    free(rows_recurse);
    free(cols_output_reduce);
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
            double val = cimj(reduce_i, rows[i], cols[z]);
            if (val < best) {
                best = val;
                best_idx = z;
            }
        }
#ifdef DEBUG
        printf("[mincompute] %d: cols[best_idx] = %d\n", i, cols[best_idx]);
#endif
        cols_output[i] = cols[best_idx];
        first = end;
    }
    for (size_t i = 0; i < n; i+=2) cols_output[i] = output[i/2];
    free(output);
#ifdef DEBUG
    printf("[mincompute] reduce_i = %d   n = %d\n", reduce_i, n);
    for (size_t i = 0; i < n; ++i) {
        printf("[mincompute] cols_output[%d] = %d\n", i, cols_output[i]);
    }
    printf("[mincompute] returning\n");
#endif
    return;
}

static void fill_row2(size_t k) {
    size_t n = t.n;
    size_t *rows = (size_t *) malloc(n * sizeof(size_t));
    size_t *cols = (size_t *) malloc(n * sizeof(size_t));
    size_t *output = (size_t *) malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) {
        rows[i] = i;
        cols[i] = i;
    }

#ifdef DEBUG
    printf("Matrix cimj:\n");
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%.2f  ", cimj(k, i, j));
        }
        printf("\n");
    }
    printf(" ------    fill_row2   k = %ld   ---------\n", k);
#endif
    mincompute(rows, cols, n, n, k, output);

    for (size_t i = 0; i < n; ++i) {
#ifdef DEBUG
        printf("output[%d] = %d\n", i, output[i]);
#endif
        size_t row = i;
        size_t col = output[i];
        set(&t, k, i, cimj(k, row, col));
    }
    free(rows);
    free(cols);
    free(output);
    return;
}


static void base_case(size_t k) {
    size_t n = t.n;
    for (size_t i = 0; i < n; ++i) {
        double cost = cost_interval_l2(&ps, 0, i);
        set(&t, 1, i, cost);
    }
    set(&t, 1, 0, 0.0);

    for (size_t j = 1; j <= k; ++j) {
        set(&t, j, 0, 0.0);
    }
    return;
}


// static function not visible in other compilation units.
static double kmeans(double *points, size_t n,
                     double *centers, size_t k) {
    init_Table(&t, k + 1, n, 0.0);
    init_IntervalSum(&ps, points, n);
    base_case(k);
    evict_row(&t, 0);
    for (size_t c = 2; c <= k; ++c) {
        fill_row2(c);
        evict_row(&t, c-1);
    }
    //print_Table(&t);
    double ret = get(&t, k, n - 1);
    free_Table(&t);
    free_IntervalSum(&ps);
    return ret;
}

kmeans_fn get_kmeans_fast() {
  return kmeans;
}

