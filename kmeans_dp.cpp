#include "kmeans.h"
#include <memory>
#include <iostream>

kmeans_dp::kmeans_dp(const std::vector<double> &points) : is(points), row(points.size()), row_prev(points.size()),
                                                          points(points.begin(), points.end()), n(points.size()) {}

double kmeans_dp::report(std::vector<double> &points, size_t k, std::vector<double> &centers) {
    if (k == 1 || points.size() == 1) {
        interval_sum<double> is(points);
        centers.push_back(is.mean(0, points.size() - 1));
        return is.query(0, points.size());
    }

    std::vector<double> reversed_points(points.rbegin(), points.rend());
    for (auto &p : reversed_points) p = -p;

    std::vector<double> last_row_left;
    std::vector<double> last_row_right;


    {
        std::unique_ptr<kmeans_dp> left(get_instance(points));
        std::unique_ptr<kmeans_result> res_left(left->compute(k / 2));
        last_row_left = std::move(left->row);
    }

    {
        std::unique_ptr<kmeans_dp> right(get_instance(reversed_points));
        std::unique_ptr<kmeans_result> res_right(right->compute(k - k/2));
        last_row_right = std::move(right->row);
    }
    double best = std::numeric_limits<double>::max();
    size_t best_idx = 0;
    for (size_t i = 0; i < points.size() - 1; ++i) {
        double cost_left = last_row_left[i]; // cost of clustering points[0..i] into k/2 clusters.
        double cost_right = last_row_right[points.size() - 2 - i]; // cost of clustering points[n-1 .. i+1] into k-k/2 clusters.

        double cost = cost_left + cost_right;
        if (cost < best) {
            best = cost;
            best_idx = i;
        }
    }

    {
        std::vector<double> empty;
        std::swap(empty, last_row_left);
    }
    {
        std::vector<double> empty;
        std::swap(empty, last_row_right);
    }
    {
        std::vector<double> empty;
        std::swap(empty, reversed_points);
    }
    std::vector<double> points_left(points.begin(), points.begin() + best_idx + 1);
    std::vector<double> points_right(points.begin() + best_idx + 1, points.end());
    report(points_left, k/2, centers);
    report(points_right, k - k/2, centers);
    return best;
}

std::unique_ptr<kmeans_result> kmeans_dp::compute_and_report(size_t k) {
    std::vector<double> centers;
    report(points, k, centers);
    std::unique_ptr<kmeans_result> res = compute(k);
    res->centers = std::move(centers);
    return res;
}
