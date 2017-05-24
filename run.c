#include "math.h"
#include "stdio.h"
#include "time.h"

#include "kmeans.h"

void correctness_test_random() {
    kmeans_fn fast = get_kmeans_fast();
    kmeans_fn slow = get_kmeans_slow();
    kmeans_fn medi = get_kmeans_medi();
    size_t n = 20;
    double points[20] = {0.0041841036041334601, 0.016864905913439476, 0.091539430201843741, 0.11167850389725253, 0.11729255208759837, 0.15870772838060987, 0.21537383129510801, 0.22030075252311732, 0.29234574608234609, 0.34182095515978905, 0.38064794144662972, 0.42369328807073692, 0.42898263636024347, 0.46299304217492687, 0.59849854723755469, 0.77144917504818644, 0.78318033400636167, 0.8393332644552387, 0.92763049366511063, 0.98685245969033264};
    for (size_t k = 1; k < 10; ++k) {
        double fast_res = fast(points, n, 0, k);
        double slow_res = slow(points, n, 0, k);
        double medi_res = medi(points, n, 0, k);
        if (fast_res != slow_res || fast_res != medi_res) {
            printf("Test failed    k=%ld\n", k);
        }
        double diff_fast_slow = abs(slow_res - fast_res);
        double diff_fast_medi = abs(medi_res - fast_res);
        printf("fast_res: %.10f\nmedi_res: %.10f\nslow_res: %.10f\n", fast_res, medi_res, slow_res);
        printf("|slow_res-fast_res|: %.10f\n|medi_res-fast_res|: %.10f\n", diff_fast_slow, diff_fast_medi);
    }
}

void timing_tests() {
    kmeans_fn fast = get_kmeans_fast();
    kmeans_fn medi = get_kmeans_medi();
    static size_t max_n = (1<<26);
    double *points = (double *) malloc(max_n * sizeof(double));
    for (size_t i = 0; i < max_n; ++i) {
        points[i] = i;
    }
    FILE *f = fopen("timing.csv", "a+");
    fprintf(f, "n, k, time_fast, time_medi\n");
    fclose(f);
    for (size_t n = 256; n <= max_n; n *= 2) {
        for (size_t k = 2; k <= 128; k *= 2) {
            clock_t before = clock();
            double fast_res = fast(points, n, 0, k);
            clock_t between = clock();
            double medi_res = medi(points, n, 0, k);
            clock_t after = clock();
            clock_t time_fast = between - before;
            clock_t time_medi = after - between;
            double diff_fast_medi = abs(medi_res - fast_res);
            if (fast_res != medi_res) {
                fprintf(stderr, "Error\n");
                fprintf(stderr, "diff = %.10f\n\n", diff_fast_medi);
            }
            FILE *f = fopen("timing.csv", "a+");
            fprintf(f, "%ld, %ld, %ld, %ld\n", n, k, time_fast, time_medi);
            fflush(f);
            fclose(f);
        }
    }
    free(points);
}


void reporting_test() {
    size_t n = 20;
    //double points[20] = {0.0041841036041334601, 0.016864905913439476, 0.091539430201843741, 0.11167850389725253, 0.11729255208759837, 0.15870772838060987, 0.21537383129510801, 0.22030075252311732, 0.29234574608234609, 0.34182095515978905, 0.38064794144662972, 0.42369328807073692, 0.42898263636024347, 0.46299304217492687, 0.59849854723755469, 0.77144917504818644, 0.78318033400636167, 0.8393332644552387, 0.92763049366511063, 0.98685245969033264};
    double *points = (double *) malloc(20*sizeof(double));
    for (size_t i = 0; i < 20; ++i) {
        points[i] = i+1;
    }
    kmeans_fn fast = get_kmeans_fast();
    double *f = (double *) malloc(20*sizeof(double));
    for (size_t k = 2; k <= 10; ++k) {
        double *centers = (double *) malloc(k * sizeof(double));
        double cost_report = report_clusters(points, n, centers, k, fast);
        double cost_actual = fast(points, n, f, k);
        double diff = cost_report - cost_actual;
        printf("cost_report=%.5f     cost_actual=%.5f    diff=%.5f\n", cost_report, cost_actual, diff);
        printf("centers: ");
        for (size_t i = 0; i < k; ++i) {
            printf("%.5f  ", centers[i]);
        }
        printf("\n");
        free(centers);
    }
    free(points);
    free(f);
    return;
}

int main() {

    /* return 0; */
    /* reporting_test(); */
    /* return 0; */
    /* timing_tests(); */
    /* return 0; */
    /* correctness_test_random(); */
    /* return 0; */
    kmeans_fn fast = get_kmeans_fast();
    kmeans_fn slow = get_kmeans_slow();
    kmeans_fn lloy = get_kmeans_lloyd();
    double points[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double last_row[9] = {0,0,0,0,0,0,0,0,0};
    size_t n = 9;
    size_t k = 3;
    double val_fast = fast(points, n, last_row, k);
    double val_slow = slow(points, n, 0, k);

    if (val_fast != val_slow) {
        printf("Error!\n");
    }
    printf("slow: %.2f\nfast: %.2f\n", val_slow, val_fast);
    double val_lloyd = lloy(points, n, 0, k);
    printf("lloyds: %.2f\n", val_lloyd);
    return 0;

}
