#include <iomanip>
#include <iostream>
#include <tuple>

#include "kmeans.h"
#include "common.h"

using namespace std;

void correctness_test_random() {
    bool fail = false;
    kmeans_fn fast = get_kmeans_fast();
    kmeans_fn slow = get_kmeans_slow();
    kmeans_fn medi = get_kmeans_medi();
    kmeans_fn hirc = get_kmeans_hirsch_larmore();
    size_t n = 20;
    double points[20] = {0.0041841036041334601, 0.016864905913439476,
                         0.091539430201843741, 0.11167850389725253,
                         0.11729255208759837, 0.15870772838060987,
                         0.21537383129510801, 0.22030075252311732,
                         0.29234574608234609, 0.34182095515978905,
                         0.38064794144662972, 0.42369328807073692,
                         0.42898263636024347, 0.46299304217492687,
                         0.59849854723755469, 0.77144917504818644,
                         0.78318033400636167, 0.8393332644552387,
                         0.92763049366511063, 0.98685245969033264};
    printf("Running tests for fast, slow and medi algorithms.\n");
    for (size_t k = 1; k < 10; ++k) {
        set_seed(0);
        double fast_res = fast(points, n, 0, k);
        set_seed(0);
        double slow_res = slow(points, n, 0, k);
        set_seed(0);
        double medi_res = medi(points, n, 0, k);
        set_seed(0);
        double hirc_res = hirc(points, n, 0, k);
        if (fast_res != slow_res) {
            printf("Test failed    k=%ld\n", k);
            printf("    fast and slow disagree on cost.");
            printf(" (fast: %.10f   slow: %.10f)\n", fast_res, slow_res);
            fail = true;
        }
        if (fast_res != medi_res) {
            printf("Test failed    k=%ld\n", k);
            printf("    fast and medi disagree on cost.");
            printf("(fast: %.10f   medi: %.10f)\n", fast_res, medi_res);
            fail = true;
        }
        if (fast_res != hirc_res) {
            printf("Test failed    k=%ld\n", k);
            printf("    fast and hirc disagree on cost.");
            printf("(fast: %.10f   hirc: %.10f)\n", fast_res, hirc_res);
            fail = true;
        }
        double diff_fast_slow = abs(slow_res - fast_res);
        double diff_fast_medi = abs(medi_res - fast_res);
        double diff_fast_hirc = abs(hirc_res - fast_res);
        if (fail) {
            printf("fast_res: %.10f\nmedi_res: %.10f\nslow_res: %.10f\n",
                   fast_res, medi_res, slow_res);
            printf("|slow_res-fast_res|: %.10f\n|medi_res-fast_res|: %.10f\n",
                   diff_fast_slow, diff_fast_medi);
            printf("|hirc_res-fast_res|: %.10f\n", diff_fast_hirc);

        }
    }
    if (fail) {
        cout << "Tests failed for fast, slow, medi, and hirc algorithm." << endl;
    } else {
        cout << "Tests succeeded for fast, slow, medi, and hirc algorithm." << endl;
    }
}

void correctness_lloyd() {
    cout << "Running tests for Lloyds algorithm" << endl;
    bool fail = false;
    kmeans_fn lloyd = get_kmeans_lloyd();
    kmeans_fn lloyd_slow = get_kmeans_lloyd_slow();
    double points[20] = {0.0041841036041334601, 0.016864905913439476,
                         0.091539430201843741, 0.11167850389725253,
                         0.11729255208759837, 0.15870772838060987,
                         0.21537383129510801, 0.22030075252311732,
                         0.29234574608234609, 0.34182095515978905,
                         0.38064794144662972, 0.42369328807073692,
                         0.42898263636024347, 0.46299304217492687,
                         0.59849854723755469, 0.77144917504818644,
                         0.78318033400636167, 0.8393332644552387,
                         0.92763049366511063, 0.98685245969033264};

    for (size_t seed = 13; seed < 13*13; seed += 13) {
        for (size_t k = 1; k <= 10; ++k) {
            set_seed(seed);
            double cost = lloyd(points, 20, 0, 3);
            set_seed(seed);
            double cost_slow = lloyd_slow(points, 20, 0, 3);
            if (cost != cost_slow) {
                cout << "Llloyd Error for k = " << k << "  seed = " << seed << endl;
                cout << "  cost_slow: " << cost_slow << "   cost: " << cost << endl;
                fail = true;
            }
        }
    }
    if (fail) {
        cout << "Tests failed for Lloyds algorithm" << endl;
    } else {
        cout << "Tests succeeded for Lloyds algorithm" << endl;
    }
}

void small_test_hirschberg_larmore() {
    double points[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    kmeans_fn hirsch = get_kmeans_hirsch_larmore();
    hirsch(points, 6, 0, 2);
    kmeans_fn slow = get_kmeans_slow();
    double res = slow(points, 6, 0, 2);
    cout << "slow: " << res << endl;
}

void another() {
    double points[20] = {0.0041841036041334601, 0.016864905913439476,
                         0.091539430201843741, 0.11167850389725253,
                         0.11729255208759837, 0.15870772838060987,
                         0.21537383129510801, 0.22030075252311732,
                         0.29234574608234609, 0.34182095515978905,
                         0.38064794144662972, 0.42369328807073692,
                         0.42898263636024347, 0.46299304217492687,
                         0.59849854723755469, 0.77144917504818644,
                         0.78318033400636167, 0.8393332644552387,
                         0.92763049366511063, 0.98685245969033264};
    double last_row[20] = {0,0,0,0,0,
                           0,0,0,0,0,
                           0,0,0,0,0,
                           0,0,0,0,0};
    size_t k = 6;
    kmeans_fn hirsch = get_kmeans_hirsch_larmore();
    kmeans_fn fast = get_kmeans_fast();
    double val_h = hirsch(points, 20, 0, k);
#ifdef DEBUG
    cout << "running fast algorithm" << endl;
#endif
    double val_f = fast(points, 20, last_row, k);
    double diff = abs(val_h - val_f);
#ifdef DEBUG
    cout << "running report clusters algorithm" << endl;
#endif
    report_clusters(points, 20, last_row, k, fast);
    for (size_t i = 0 ; i < k; ++i) {
        cout << last_row[i] << ", ";
    }
    cout << endl;
    cout << "hirsch: " << val_h << endl
         << "fast  : " << val_f << endl
         << "diff  : " << setprecision(10) << diff << endl;
}

void small() {
    double points[] = {1.0, 2.0, 3e300, 4e300};
    size_t n = 2;
    size_t k = 2;
    kmeans_fn hirsch = get_kmeans_hirsch_larmore();
    kmeans_fn fast = get_kmeans_fast();
    double hirsch_val = hirsch(points, n, 0, k);
    double fast_val = fast(points, n, 0, k);
    if (hirsch_val != fast_val) cout << "error" << endl;
    cout << hirsch_val << "   " << fast_val << endl;
}

int main(int argc, char *argv[]) {
    if (argc == 1 || argv[1] == std::string("all")) {
        correctness_test_random();
        correctness_lloyd();
        //small_test_hirschberg_larmore();
        return 0;
    }
    if (argc == 1) {
        return 1;
    }
    if (argv[1] == std::string("hirsch")) {
        //another();
        small();
    }
}
