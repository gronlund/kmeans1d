#include <iomanip>
#include <iostream>
#include <tuple>
#include <typeinfo>
#include <sstream>

#include "kmeans.h"

using namespace std;

double get_cost(kmeans &km, size_t k) {
    std::unique_ptr<kmeans_result> res(km.compute_and_report(k));
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

    std::unique_ptr<kmeans> linear(new kmeans_linear(points));
    std::unique_ptr<kmeans> slow(new kmeans_slow(points));
    std::unique_ptr<kmeans> medi(new kmeans_medi(points));
    std::unique_ptr<kmeans> wilber(new kmeans_wilber(points));

    cout << "Running tests for linear, slow and medi algorithms." << endl;
    bool any_fail = false;
    for (size_t k = 1; k < 10; ++k) {
        double linear_res = get_cost(*linear, k);
        double slow_res = get_cost(*slow, k);
        double medi_res = get_cost(*medi, k);
        double wilber_res = get_cost(*wilber, k);
        bool fail = false;
        if (linear_res != slow_res) {
            cout << "[k = " << k << "] Test failed for linear" << endl;
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
        if (slow_res != wilber_res) {
            cout << "[k = " << k << "] Test failed for wilber" << endl;
            fail = true;
        } else {
            cout << "[k = " << k << "] Test succeeded for wilber" << endl;
        }
        if (fail) {
            any_fail = true;
            double diff_linear_slow = abs(linear_res - slow_res);
            double diff_medi_slow = abs(medi_res - slow_res);
            double diff_wilber_slow = abs(wilber_res - slow_res);

            cout << "[k = " << k << "] Failed" << endl;
            cout << "Additional info below:" << endl;
            cout << "[k = " << k << "] ";
            cout << "|linear - slow| = " << setprecision(10) << diff_linear_slow << endl;

            cout << "[k = " << k << "] ";
            cout << "|medi - slow| = " << setprecision(10) << diff_medi_slow << endl;

            cout << "[k = " << k << "] ";
            cout << "|wilber - slow| = " << setprecision(10) << diff_wilber_slow << endl;
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

double get_cost(std::unique_ptr<kmeans_result> &res, std::vector<double> &points, size_t k) {
    double computed_cost = 0;
    for (auto point : points) {
        double closest = res->centers[0];
        for (auto center : res->centers) {
            if (abs(point - center) < abs(point - closest)) closest = center;
        }
        computed_cost += (point-closest) * (point - closest);
    }
    return computed_cost;
}

bool check_clustering_size(std::unique_ptr<kmeans_result> &res, std::vector<double> &points, size_t k) {
    if (res->centers.size() != k || res->centers.size() < 1) return false;
    return true;
}

bool check_clustering_cost(std::unique_ptr<kmeans_result> &res, std::vector<double> &points, size_t k) {
    double computed_cost = get_cost(res, points, k);
    if (abs(computed_cost - res->cost) > 1e-6) return false;
    return true;
}

void test_cluster_cost_equal_returned_cost() {
    std::string prefix_template = "[test_cluster_cost_equal_returned_cost] ";
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
    std::unique_ptr<kmeans> linear(new kmeans_linear(points));
    std::unique_ptr<kmeans> slow(new kmeans_slow(points));
    std::unique_ptr<kmeans> medi(new kmeans_medi(points));
    std::unique_ptr<kmeans> wilber(new kmeans_wilber(points));
    bool fail = false;
    for (size_t k = 1; k < 10; ++k) {
        std::unique_ptr<kmeans_result> linear_res = linear->compute_and_report(k);
        std::unique_ptr<kmeans_result> medi_res = medi->compute_and_report(k);
        std::unique_ptr<kmeans_result> slow_res = slow->compute_and_report(k);
        std::unique_ptr<kmeans_result> wilber_res = wilber->compute_and_report(k);
        std::stringstream ss;
        ss << prefix_template << "[ k = " << k << " ] ";
        std::string prefix = ss.str();
        if (!check_clustering_size(linear_res, points, k)) {
            cout << prefix << "linear clustering expected size " << k
                 << " found size " << linear_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(linear_res, points, k)) {
            cout << prefix << "linear clustering cost failed. Returned cost " << linear_res->cost
                 << "  Computed cost " << get_cost(linear_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(medi_res, points, k)) {
            cout << prefix << "medi clustering expected size " << k
                 << " found size " << medi_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(medi_res, points, k)) {
            cout << prefix << "medi clustering cost failed. Returned cost " << medi_res->cost
                 << "  Computed cost " << get_cost(medi_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(slow_res, points, k)) {
            cout << prefix << "slow clustering expected size " << k
                 << " found size " << slow_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(slow_res, points, k)) {
            cout << prefix << "slow clustering cost failed. Returned cost " << slow_res->cost
                 << "  Computed cost " << get_cost(slow_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(wilber_res, points, k)) {
            cout << prefix << "wilber clustering expected size " << k
                 << " found size " << wilber_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(wilber_res, points, k)) {
            cout << prefix << "wilber clustering cost failed. Returned cost " << wilber_res->cost
                 << "  Computed cost " << get_cost(wilber_res, points, k) << endl;
            fail = true;
        }
    }
    if (fail) {
        cout << prefix_template << "Failed." << endl;
    } else {
        cout << prefix_template << "Succeeded." << endl;
    }
}

void more_clusters_than_points() {
    // TODO: test it.
    std::vector<double> points = {1.0, 2.0, 3.0, 4.0};
    size_t k = 10;
    std::vector<std::shared_ptr<kmeans> > algs = {
        std::shared_ptr<kmeans>(new kmeans_linear(points)),
        std::shared_ptr<kmeans>(new kmeans_slow(points)),
        std::shared_ptr<kmeans>(new kmeans_medi(points)),
        std::shared_ptr<kmeans>(new kmeans_wilber(points)),
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
        test_cluster_cost_equal_returned_cost();
        return 0;
    }
    if (argc == 1) {
        return 1;
    }
    return 0;
}
