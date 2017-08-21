#include "interval_sum.hpp"
#include <vector>
#include <cstdint>
#include <iostream>

template<typename T>
interval_sum<T>::interval_sum(const std::vector<T> &points) : prefix_sum(points.size() + 1),
                                                              prefix_sum_of_squares(points.size() + 1) {
    T prefix_sum_val = 0;
    T prefix_square_val = 0;
    size_t n = points.size();
    this->prefix_sum[0] = 0;
    this->prefix_sum_of_squares[0] = 0;
    for (size_t i = 1; i <= n; ++i) {
        prefix_sum_val += points[i-1];
        prefix_square_val += points[i-1]*points[i-1];
        this->prefix_sum[i] = prefix_sum_val;
        this->prefix_sum_of_squares[i] = prefix_square_val;
    }
}

template<typename T>
interval_sum<T>::interval_sum() {}

template<typename T>
interval_sum<T>::~interval_sum() {}

/**
 * @returns sum from i to j of points[l]^2 where i included, j not included.
 */
template<typename T>
T interval_sum<T>::query_sq(size_t i, size_t j) const {
    return this->prefix_sum_of_squares[j] - this->prefix_sum_of_squares[i];
}

/**
 * @returns sum from i to j of points[l] where i included, j not included.
 */
template<typename T>
T interval_sum<T>::query(size_t i, size_t j) const {
    return this->prefix_sum[j] - this->prefix_sum[i];
}

/**
 * returns the cost of clustering points [start, i] into one cluster with L_2
 * norm.
 */
template<typename T>
T interval_sum<T>::cost_interval_l2(size_t start, size_t i) const {
    //std::cout << "start = " << start << "    i = " << i << std::endl;
    T suffix_sum = this->query(start, i+1);
    T suffix_sq = this->query_sq(start, i+1);
    //std::cout << "suffix_sum    = " << suffix_sum << std::endl;
    //std::cout << "suffix_sum_sq = " << suffix_sq << std::endl;
    size_t length = i-start+1;
    //std::cout << "length        = " << length << std::endl;
    T mean = suffix_sum / length;
    //std::cout << "mean          = " << mean << std::endl;
    T interval_cost = suffix_sq + mean*mean*length - 2 * suffix_sum * mean;
    //std::cout << "interval_cost = " << interval_cost << std::endl;
    return interval_cost;
}

template class interval_sum<double>;
template class interval_sum<std::int64_t>;
