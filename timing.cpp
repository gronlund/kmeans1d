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
static size_t start = 10000000;
static size_t increment = 10000000;

static omp_lock_t lock;

vector<double> generate(size_t n) {
    mt19937 mt(time(0));
    vector<double> res(n, 0);
    double maximum = 1;
    for (size_t i = 0; i < n; ++i) {
        res[i] = mt();
        maximum = std::max(maximum, res[i]);
    }
    std::sort(res.begin(), res.end());
    for (auto &v : res) {
        v = v / maximum;
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

int main(int argc, char *argv[]) {

    //vector<size_t> ks = {1, 10, 50, 100, 500};
    vector<size_t> ks = {10, 20};
    {
        ofstream f(datafile_name, ios_base::out);
        f << "n,k,linear,medi,linear_report,medi_report,lloyd_report,wilber" << std::endl;
    }
    omp_init_lock(&lock);
    for (size_t n = start; ; n += increment) {
        vector<double> points = generate(n);

        for (size_t i = 0; i < ks.size(); ++i) {
            size_t k = ks[i];
            std::chrono::milliseconds linear_time, medi_time, linear_time_report, medi_time_report;
            std::chrono::milliseconds lloyd_time_report, wilber_time;

#pragma omp parallel for
            for (size_t alg = 0; alg < 6; ++alg) {
                switch (alg) {
                case 0:
                    linear_time = time_compute<kmeans_linear>(points, k);
                    break;
                case 1:
                    medi_time = time_compute<kmeans_medi>(points, k);
                    break;
                case 2:
                    linear_time_report = time_compute_and_report<kmeans_linear>(points, k);
                    break;
                case 3:
                    medi_time_report = time_compute_and_report<kmeans_medi>(points, k);
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
                  << medi_time.count() << ","
                  << linear_time_report.count() << ","
                  << medi_time_report.count() << ","
                  << lloyd_time_report.count() << ","
                  << wilber_time.count() << endl;
            }
            omp_unset_lock(&lock);
        }

    }
    return 0;
}
