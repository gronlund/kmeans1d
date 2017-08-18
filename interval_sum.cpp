#include "interval_sum.hpp"
#include <vector>
#include <cstdint>

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

template<typename T>
T interval_sum<T>::query_sq(size_t i, size_t j) const {
    return this->prefix_sum_of_squares[j] - this->prefix_sum_of_squares[i];
}

template<typename T>
T interval_sum<T>::query(size_t i, size_t j) const {
    return this->prefix_sum[j] - this->prefix_sum[i];
}

template<typename T>
T interval_sum<T>::cost_interval_l2(size_t start, size_t i) const {
    T suffix_sum = this->query(start, i+1);
    T suffix_sq = this->query_sq(start, i+1);
    size_t length = i-start+1;
    T mean = suffix_sum / length;
    T interval_cost = suffix_sq + mean*mean*length - 2 * suffix_sum * mean;
    return interval_cost;
}

template class interval_sum<double>;
template class interval_sum<std::int64_t>;
