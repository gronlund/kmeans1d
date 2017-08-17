#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <stdlib.h>
#include <stdio.h>

typedef double (*kmeans_fn)(double *points, size_t n,
                            double *last_row, size_t k);


kmeans_fn get_kmeans_slow();

kmeans_fn get_kmeans_medi();

kmeans_fn get_kmeans_fast();

kmeans_fn get_kmeans_lloyd();

kmeans_fn get_kmeans_lloyd_slow();

kmeans_fn get_kmeans_hirsch_larmore();

double report_clusters(double *points, size_t n,
                       double *centers, size_t k,
                       kmeans_fn);

#endif /* __KMEANS_H__ */
