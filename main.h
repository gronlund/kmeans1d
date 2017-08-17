#include <cstdlib>
#include <vector>

struct Table {
    size_t k, n;
    std::vector<double> arr;

    Table(size_t k, size_t n, double init_val);

    double get(size_t i, size_t j);

    void set(size_t i, size_t j, double val);
};

struct IntervalSum {
    double *prefix_sum;
    double *prefix_sum_of_squares;

    IntervalSum(const double *points, size_t n);

    /**
     * @returns sum from i to j of points[l]^2 where i included, j not included.
     */
    double query_sq(size_t i, size_t j);

    /**
     * @returns sum from i to j of points[l] where i included, j not included.
     */
    double query(size_t i, size_t j);

    ~IntervalSum();

};

void kmeans1d_solve(double *points, size_t n, size_t k,
		    Table &dp_table, IntervalSum &IS, int norm);

void kmeans_cluster_centers(double *points, size_t n, size_t k, Table &dp_table,
			    IntervalSum &IS, double *clusters, int norm);
