#include <iomanip>
#include <iostream>
#include <tuple>
#include <typeinfo>

#include "kmeans.h"
#include "common.h"

using namespace std;

double get_cost(kmeans &km, size_t k) {
    std::unique_ptr<kmeans_result> res(km.compute(k));
    return res->cost;
}

void correctness_test_random() {
    std::vector<double> points = {0.0041841036041334601, 0.016864905913439476,
                                  0.091539430201843741, 0.11167850389725253,
                                  0.11729255208759837, 0.15870772838060987,
                                  0.21537383129510801, 0.22030075252311732,
                                  0.29234574608234609, 0.34182095515978905,
                                  0.38064794144662972, 0.42369328807073692,
                                  0.42898263636024347, 0.46299304217492687,
                                  0.59849854723755469, 0.77144917504818644,
                                  0.78318033400636167, 0.8393332644552387,
                                  0.92763049366511063, 0.98685245969033264};

    std::unique_ptr<kmeans> fast(new kmeans_fast(points));
    std::unique_ptr<kmeans> slow(new kmeans_slow(points));
    std::unique_ptr<kmeans> medi(new kmeans_medi(points));
    std::unique_ptr<kmeans> hirc(new kmeans_hirschberg_larmore(points));

    printf("Running tests for fast, slow and medi algorithms.\n");
    bool any_fail = false;
    for (size_t k = 1; k < 10; ++k) {
        double fast_res = get_cost(*fast, k);
        double slow_res = get_cost(*slow, k);
        double medi_res = get_cost(*medi, k);
        double hirc_res = get_cost(*hirc, k);
        bool fail = false;
        if (fast_res != slow_res) {
            cout << "[k = " << k << "] Test failed for fast" << endl;
            fail = true;
        } else {
            cout << "[k = " << k << "] Test succeeded for slow" << endl;
        }
        if (slow_res != medi_res) {
            cout << "[k = " << k << "] Test failed for medi" << endl;
            fail = true;
        } else {
            cout << "[k = " << k << "] Test succeeded for medi" << endl;
        }
        if (slow_res != hirc_res) {
            cout << "[k = " << k << "] Test failed for hirc" << endl;
            fail = true;
        } else {
            cout << "[k = " << k << "] Test succeeded for hirc" << endl;
        }
        if (fail) {
            any_fail = true;
            double diff_fast_slow = abs(fast_res - slow_res);
            double diff_medi_slow = abs(medi_res - slow_res);
            double diff_hirc_slow = abs(hirc_res - slow_res);

            cout << "[k = " << k << "] Failed" << endl;
            cout << "Additional info below:" << endl;
            cout << "[k = " << k << "] ";
            cout << "|fast - slow| = " << setprecision(10) << diff_fast_slow << endl;

            cout << "[k = " << k << "] ";
            cout << "|medi - slow| = " << setprecision(10) << diff_medi_slow << endl;

            cout << "[k = " << k << "] ";
            cout << "|hirc - slow| = " << setprecision(10) << diff_hirc_slow << endl;
        }
    }
    if (any_fail) {
        cout << "[correctness_test_random] Failed" << endl;
    } else {
        cout << "[correctness_test_random] Succeeded" << endl;
    }
}

void correctness_lloyd() {
    cout << "Running tests for Lloyds algorithm" << endl;
    bool fail = false;
    std::vector<double> points = {0.0041841036041334601, 0.016864905913439476,
                                  0.091539430201843741, 0.11167850389725253,
                                  0.11729255208759837, 0.15870772838060987,
                                  0.21537383129510801, 0.22030075252311732,
                                  0.29234574608234609, 0.34182095515978905,
                                  0.38064794144662972, 0.42369328807073692,
                                  0.42898263636024347, 0.46299304217492687,
                                  0.59849854723755469, 0.77144917504818644,
                                  0.78318033400636167, 0.8393332644552387,
                                  0.92763049366511063, 0.98685245969033264};

    std::unique_ptr<kmeans_lloyd> lloyd_slow(new kmeans_lloyd_slow(points));
    std::unique_ptr<kmeans_lloyd> lloyd_fast(new kmeans_lloyd_fast(points));

    for (size_t seed = 13; seed < 13*13; seed += 13) {
        for (size_t k = 1; k <= 10; ++k) {
            lloyd_fast->set_seed(seed);
            double cost_fast = get_cost(*lloyd_fast, k);
            lloyd_slow->set_seed(seed);
            double cost_slow = get_cost(*lloyd_slow, k);
            if (cost_fast < 0) {
                fail = true;
                cout << "[lloyd] Error [fast], negative cost: " << cost_fast << std::endl;
            }
            if (cost_slow < 0) {
                cout << "[lloyd] Error [slow], negative cost: " << cost_slow << std::endl;
            }
            if (cost_fast != cost_slow) {
                cout << "Lloyd Error for k = " << k << "  seed = " << seed << endl;
                cout << "  cost_slow: " << cost_slow << "   cost: " << cost_fast << endl;
                fail = true;
            }
        }
    }
    if (fail) {
        cout << "[correctness_lloyd] Failed." << endl;
    } else {
        cout << "[correctness_lloyd] Succeeded." << endl;
    }
}

void more_clusters_than_points() {
    // TODO: test it.
    std::vector<double> points = {1.0, 2.0, 3.0, 4.0};
    size_t k = 10;
    std::vector<std::shared_ptr<kmeans> > algs = {
        std::shared_ptr<kmeans>(new kmeans_fast(points)),
        std::shared_ptr<kmeans>(new kmeans_slow(points)),
        std::shared_ptr<kmeans>(new kmeans_medi(points)),
        std::shared_ptr<kmeans>(new kmeans_hirschberg_larmore(points)),
        std::shared_ptr<kmeans>(new kmeans_lloyd_slow(points)),
        std::shared_ptr<kmeans>(new kmeans_lloyd_fast(points))
    };
    bool fail = false;
    for (size_t i = 0; i < algs.size(); ++i) {
        double cost = get_cost(*algs[i], k);
        if (cost >= 1e-6) {
            std::cout << "[more_clusters_than_points] " << typeid(algs[i]).name() << " was not 0." << std::endl;
            fail = true;
        }
    }
    if (fail) {
        std::cout << "[more_clusters_than_points] Failed" << std::endl;
        return;
    }
    std::cout << "[more_clusters_than_points] Succeeded" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc == 1 || argv[1] == std::string("all")) {
        correctness_test_random();
        correctness_lloyd();
        more_clusters_than_points();
        return 0;
    }
    if (argc == 1) {
        return 1;
    }
    return 0;
}
