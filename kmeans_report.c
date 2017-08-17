#include "kmeans.h"
#include "float.h"
#include "assert.h"
#include "common.h"

static double oo = DBL_MAX;

struct List {
    double *elements;
    size_t length;
    size_t idx;
};

static void append(struct List *l, double element) {
    assert(l->idx < l->length);
    l->elements[l->idx++] = element;
}

static void init(struct List *l, size_t n) {
    l->elements = (double *) malloc(n * sizeof(double));
    l->idx = 0;
    l->length = n;
    for (size_t i = 0; i < n; ++i) l->elements[i] = 0;
    return;
}

static void free_list(struct List *l) {
    free(l->elements);
}


static double report_clusters_rec(double *points, size_t n,
                                  struct List *l, size_t k,
                                  kmeans_fn kmeans_function) {
    if (k == 1 || n == 1) {
        double val = mean(points, n);
        append(l, val);
        return cost_l2(points, n);
    }
    double *reversed_points = (double *) malloc(n * sizeof(double));
    double *last_row_points = (double *) malloc(n * sizeof(double));
    double *last_row_revers = (double *) malloc(n * sizeof(double));

    for (size_t i = 0; i < n; ++i) {
        reversed_points[i] = -points[n-1-i];
    }

    kmeans_function(points, n, last_row_points, k / 2);
    kmeans_function(reversed_points, n, last_row_revers, k - k/2);
    free(reversed_points);


    double best = oo;
    size_t best_idx = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        // cost of clustering points[0..i] into k/2 clusters
        double cost_points = last_row_points[i];

        // cost of clustering points[n-1..i+1] into k - k/2 clusters.
        double cost_revers = last_row_revers[n-2-i];
        double cost = cost_points + cost_revers;
        if (cost < best) {
            best = cost;
            best_idx = i;
        }
    }
#ifdef DEBUG
    printf("n=%ld  k=%ld  best split: %ld  cost: %.3f\n", n, k, best_idx, best);
#endif

    free(last_row_points);
    free(last_row_revers);
    report_clusters_rec(points, best_idx + 1,
                        l, k/2, kmeans_function);

    report_clusters_rec(points + best_idx + 1, n - best_idx - 1,
                        l, k - k/2,
                        kmeans_function);
    return best;
}

double report_clusters(double *points, size_t n,
                     double *centers, size_t k,
                     kmeans_fn kmeans_function) {

    struct List l;
    init(&l, k);
    double res = report_clusters_rec(points, n, &l, k, kmeans_function);
    for (size_t i = 0; i < k; ++i) {
        centers[i] = l.elements[i];
    }
    free_list(&l);
    return res;
}
