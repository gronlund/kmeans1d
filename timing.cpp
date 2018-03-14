#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "omp.h"

#include "kmeans.h"

using namespace std;

static std::string datafile_name = "data.csv";
static size_t start = 1000000;
static size_t increment = 10000000;

static omp_lock_t lock;

class input_generator {
public:
    explicit input_generator() {}
    input_generator(const input_generator&) = delete;
    input_generator& operator=(const input_generator&) = delete;
    ~input_generator() = default;

    virtual vector<double> generate(size_t n) = 0;
};

class input_generator_uniform : public input_generator {
public:
    explicit input_generator_uniform() {}
    input_generator_uniform(const input_generator_uniform&) = delete;
    input_generator_uniform& operator=(const input_generator_uniform&) = delete;
    ~input_generator_uniform() = default;
    vector<double> generate(size_t n) override {
        mt19937 mt(time(0));
        vector<double> res(n, 0);
        double factor = 1e6;
        double maximum = 0;
        for (size_t i = 0; i < n; ++i) {
            res[i] = mt();
            maximum = std::max(maximum, res[i]);
        }
        std::sort(res.begin(), res.end());
        for (auto &v : res) {
            v = (v / maximum) * factor;
        }
        return res;
    }
};

class input_generator_gauss_mixture : public input_generator {
public:
    explicit input_generator_gauss_mixture() {}
    input_generator_gauss_mixture(const input_generator_gauss_mixture&) = delete;
    input_generator_gauss_mixture& operator=(const input_generator_gauss_mixture&) = delete;
    ~input_generator_gauss_mixture() = default;
    vector<double> generate(size_t n) override {
        size_t k = 16;
        std::vector<double> centers(k, 0);
        for (size_t i = 0; i < k; ++i) {
            centers[i] = i * 1e6;
        }
        mt19937 mt(time(0));
        std::normal_distribution<double> gauss(0.0, 100.0);
        vector<double> res(n, 0);
        double factor = 1e6;
        double maximum = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t cluster = mt() % k;
            res[i] = centers[cluster] + gauss(mt);
            maximum = std::max(maximum, res[i]);
        }
        std::sort(res.begin(), res.end());
        return res;
    }
};

vector<double> generate(size_t n) {
    mt19937 mt(time(0));
    vector<double> res(n, 0);
    double factor = 1e6;
    double maximum = 0;
    for (size_t i = 0; i < n; ++i) {
        res[i] = mt();
        maximum = std::max(maximum, res[i]);
    }
    std::sort(res.begin(), res.end());
    for (auto &v : res) {
        v = (v / maximum) * factor;
    }
    return res;
}

template<typename alg>
std::chrono::milliseconds time_compute(vector<double> &points, size_t k) {
    auto start = std::chrono::high_resolution_clock::now();
    unique_ptr<kmeans> f(new alg(points));
    unique_ptr<kmeans_result> res = f->compute(k);
    auto end = std::chrono::high_resolution_clock::now();
    omp_set_lock(&lock);
    std::cout << "[" << f->name() << "] "
              << "[k = " << k << "] [n = " << points.size() << "] "
              << "[cost = " << res->cost << "]" << std::endl;
    omp_unset_lock(&lock);
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

template<typename alg>
std::chrono::milliseconds time_compute_and_report(vector<double> &points, size_t k) {
    auto start = std::chrono::high_resolution_clock::now();
    unique_ptr<kmeans> f(new alg(points));
    unique_ptr<kmeans_result> res = f->compute_and_report(k);
    auto end = std::chrono::high_resolution_clock::now();
    omp_set_lock(&lock);
    std::cout << "[" << f->name() << "] "
              << "[k = " << k << "] [n = " << points.size() << "] "
              << "[cost = " << res->cost << "]" << std::endl;
    omp_unset_lock(&lock);
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

int run(std::unique_ptr<input_generator> const &g, std::string outfilename) {

    //vector<size_t> ks = {1, 10, 50, 100, 500};
    vector<size_t> ks = {10, 20};
    {
        ofstream f(datafile_name, ios_base::out);
        f << "n,k,dp-linear,dp-monotone,dp-linear-hirsch,dp-monotone-hirsch,lloyd_report,wilber" << std::endl;
    }
    omp_init_lock(&lock);
    for (size_t n = start; ; n += increment) {
        vector<double> points = generate(n);

        for (size_t i = 0; i < ks.size(); ++i) {
            size_t k = ks[i];
            std::chrono::milliseconds linear_time, monotone_time, linear_time_report, monotone_time_report;
            std::chrono::milliseconds lloyd_time_report, wilber_time;

#pragma omp parallel for
            for (size_t alg = 0; alg < 6; ++alg) {
                switch (alg) {
                case 0:
                    linear_time = time_compute<kmeans_linear>(points, k);
                    break;
                case 1:
                    monotone_time = time_compute<kmeans_monotone>(points, k);
                    break;
                case 2:
                    linear_time_report = time_compute_and_report<kmeans_linear>(points, k);
                    break;
                case 3:
                    monotone_time_report = time_compute_and_report<kmeans_monotone>(points, k);
                    break;
                case 4:
                    lloyd_time_report = time_compute_and_report<kmeans_lloyd_fast>(points, k);
                    break;
                case 5:
                    wilber_time = time_compute_and_report<kmeans_wilber>(points, k);
                    break;
                }
            }
            omp_set_lock(&lock);
            {
                ofstream f(datafile_name, ios_base::app);
                f << n << "," << k << ","
                  << linear_time.count() << ","
                  << monotone_time.count() << ","
                  << linear_time_report.count() << ","
                  << monotone_time_report.count() << ","
                  << lloyd_time_report.count() << ","
                  << wilber_time.count() << endl;
            }
            omp_unset_lock(&lock);
        }

    }
    return 0;
}

int main(int argc, char* argv[]) {
    std::unique_ptr<input_generator> generator(nullptr);
    std::string outfilename;
    if (argc > 1) {
        if (argv[1] == "uniform") {
            generator = std::move(std::unique_ptr<input_generator>(new input_generator_uniform()));
            outfilename = datafile_name;
        } else if (argv[1] == "gauss-mixture") {
            generator = std::move(std::unique_ptr<input_generator>(new input_generator_gauss_mixture()));
            outfilename = "timings_gauss.csv";
        }
    }
    if (generator == nullptr) {
        generator = std::unique_ptr<input_generator>(new input_generator_uniform());
    }

    run(generator, datafile_name);
}
