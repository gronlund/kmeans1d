#ifndef INTERVAL_SUM_GUARD
#define INTERVAL_SUM_GUARD

#include <vector>

template<typename T>
class interval_sum {
public:
    interval_sum();
    interval_sum(const std::vector<T> &points);
    T query_sq(size_t i, size_t j) const;
    T query(size_t i, size_t j) const;
    T cost_interval_l2(size_t i, size_t j) const;
    T mean(size_t i, size_t j) const;
    ~interval_sum();
private:
    std::vector<T> prefix_sum;
    std::vector<T> prefix_sum_of_squares;
};

#endif
