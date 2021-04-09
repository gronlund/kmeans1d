#include "kmeans.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
/**
 * This code implements the divide-and-conquer algorithm
 * for computing k-means.
 */


kmeans_monotone::kmeans_monotone(const std::vector<double> &points) : kmeans_dp(points) { }
std::string kmeans_monotone::name() { return std::string("monotone"); }

std::unique_ptr<kmeans_dp> kmeans_monotone::get_instance(std::vector<double> &points) {
    return std::unique_ptr<kmeans_dp>(new kmeans_monotone(points));
}

void kmeans_monotone::fill_row_rec(size_t begin, size_t end, size_t k,
                               int64_t split_left, int64_t split_right) {

    size_t mid = (begin+end)/2;
    //if(mid == 0){
    //std::cout << "can 0 be middle " << begin << " " << end << " " << std::endl;
    //}
    double oo = std::numeric_limits<double>::max();
    double best = oo;
    int64_t best_split = mid;
    for (int64_t s = split_left; s <= split_right && s <= mid; ++s) {

        double cost_last_cluster = is.cost_interval_l2(s, mid);
        double cost_before = 0.0;
	if(s > 0){
	  cost_before = row_prev[s-1];//out of bounds on that one
	}
        double cost = oo;
	
        if (cost_before == oo || cost_last_cluster == oo) {
            cost = oo;
        } else {
            cost = cost_before + cost_last_cluster;
        }
        if (cost < best) {
            best = cost;
            best_split = s;
        }
	//assert(best >=0);
	
    }

    row[mid] = best;
    //assert(best >=0.0);
    if (mid != begin) {
        fill_row_rec(begin, mid, k, split_left, best_split);
    }
    if (mid + 1 != end) {
        fill_row_rec(mid+1, end, k, best_split, split_right);
    }
}

void kmeans_monotone::fill_row(size_t k) {
    //std::cout << "fill row " << k << std::endl;
    fill_row_rec(0, n, k, 0, n-1);
}

std::unique_ptr<kmeans_result> kmeans_monotone::compute(size_t k) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);
    for (size_t i = 1; i < n; ++i) {
        row[i] = is.cost_interval_l2(0, i);
    }
    row[0] = 0.0;

    for (size_t _k = 2; _k <= k; ++_k) {
      std::swap(row, row_prev);
      fill_row(_k);
    }
    res->cost = row[n-1];
    return res;
}
