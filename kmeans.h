#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <stdlib.h>
#include <stdio.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>

#include "interval_sum.hpp"

class kmeans_result {
public:
    double cost;
    std::vector<double> centers;
};

class kmeans {
public:
    virtual ~kmeans() {};
    virtual std::unique_ptr<kmeans_result> compute(size_t k) = 0;
    virtual std::unique_ptr<kmeans_result> compute_and_report(size_t k) = 0;
    virtual std::string name() = 0;
};

class kmeans_hirschberg_larmore : public kmeans {
public:
    kmeans_hirschberg_larmore(const std::vector<double> &points);
    std::unique_ptr<kmeans_result> compute(size_t k) override;
    std::unique_ptr<kmeans_result> compute_and_report(size_t k) override;
    ~kmeans_hirschberg_larmore() override;
    std::string name() override;
private:
    double weight(size_t i, size_t j);
    double g(size_t i, size_t j);
    bool bridge(size_t i, size_t j, size_t k, size_t n);
    std::pair<double, size_t> basic(size_t n);
    std::pair<double, size_t> traditional(size_t n);
    double get_actual_cost(size_t n, std::unique_ptr<kmeans_result> &res);

    double lambda;
    std::vector<double> f;
    std::vector<std::size_t> bestleft;
    interval_sum<double> is;
    std::vector<double> points;
    std::size_t n;
};

class kmeans_dp : public kmeans {
public:
    kmeans_dp(const std::vector<double> &points);
    virtual ~kmeans_dp() {};
    virtual std::unique_ptr<kmeans_result> compute(size_t k) = 0;
    std::unique_ptr<kmeans_result> compute_and_report(size_t k) override;
    virtual std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) = 0;
    double report(std::vector<double> &points, size_t k, std::vector<double> &centers);
protected:
    interval_sum<double> is;
    std::vector<double> row;
    std::vector<double> row_prev;
    std::vector<double> points;
    size_t n;

    //virtual std::unique_ptr<kmeans_dp> get_new_instance(std::vector<double> &points) = 0;
};

class kmeans_fast : public kmeans_dp {
public:
    kmeans_fast(const std::vector<double> &points);
    std::unique_ptr<kmeans_result> compute(size_t k) override;
    std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) override;
    std::string name() override;
private:
    double cimj(size_t i, size_t m, size_t j);
    void reduce(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
                std::vector<size_t> &cols_output, size_t reduce_i);
    void mincompute(size_t row_multiplier, std::vector<size_t> &cols, size_t n, size_t m,
                    size_t reduce_i, std::vector<size_t> &cols_output);
    void fill_row(size_t k);
    void base_case(size_t k);

};

class kmeans_medi : public kmeans_dp {
public:
    kmeans_medi(const std::vector<double> &points);
    std::unique_ptr<kmeans_result> compute(size_t k) override;
    std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) override;
    std::string name() override;
private:

    void fill_row_rec(size_t begin, size_t end, size_t k,
                      int64_t split_left, int64_t split_right);
    void fill_row(size_t k);
};

class kmeans_slow : public kmeans_dp {
public:
    kmeans_slow(const std::vector<double> &points);
    std::unique_ptr<kmeans_result> compute(size_t k) override;
    std::unique_ptr<kmeans_dp> get_instance(std::vector<double> &points) override;
    std::string name() override;
};

class kmeans_lloyd : public kmeans {
public:
    kmeans_lloyd();
    virtual std::unique_ptr<kmeans_result> compute(size_t k) = 0;
    virtual std::unique_ptr<kmeans_result> compute_and_report(size_t k) = 0;
    virtual void set_seed(std::mt19937::result_type val);
    std::mt19937::result_type random_value();
    std::vector<size_t> init_splits(size_t n, size_t k);
    virtual std::string name() override;
private:
    std::mt19937_64 mt;
};

class kmeans_lloyd_slow : public kmeans_lloyd {
public:
    kmeans_lloyd_slow(const std::vector<double> &points);
    std::unique_ptr<kmeans_result> compute(size_t k) override;
    std::unique_ptr<kmeans_result> compute_and_report(size_t k) override;
private:
    std::vector<double> points;
    interval_sum<double> is;
    size_t n;
};

class kmeans_lloyd_fast : public kmeans_lloyd {
public:
    kmeans_lloyd_fast(const std::vector<double> &points);
    std::unique_ptr<kmeans_result> compute(size_t k) override;
    std::unique_ptr<kmeans_result> compute_and_report(size_t k) override;
private:
    std::vector<double> points;
    interval_sum<double> is;
    size_t n;
};

typedef double (*kmeans_fn)(double *points, size_t n,
                            double *last_row, size_t k);


double report_clusters(double *points, size_t n,
                       double *centers, size_t k,
                       kmeans_fn);

#endif /* __KMEANS_H__ */
