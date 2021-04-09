#include <cassert>
#include <limits>
#include <iostream>
#include <iomanip> 
#include <vector>
#include <algorithm>
#include <cassert>
#include "kmeans.h"

template <typename T>
void print_vector(const std::vector<T> & vec, std::string sep=", "){
    for(auto elem : vec)
    {
        std::cout << elem << sep;
    }
    std::cout<<std::endl;
}

void kmeans_wilber::set_search_strategy(search_strategy strat) {
    std::cout << "setting search strategy to: " << strat << std::endl;
    this->search_strat = strat;
}

kmeans_wilber::kmeans_wilber(const std::vector<double> &points) :
    f(points.size() + 1, 0.0), bestleft(points.size() + 1, 0),
    is(points), points(points), n(points.size()), search_strat(search_strategy::INTERPOLATION) { }

std::string kmeans_wilber::name() { return std::string("wilber"); }

std::unique_ptr<kmeans_result> kmeans_wilber::compute_binary_search(size_t k) {
    std::unique_ptr<kmeans_result> kmeans_res(new kmeans_result);
    double lo = 0.0;
    double hi = is.cost_interval_l2(0, n-1);

    size_t cnt = 0;
    double val_found;
    size_t k_found;
    while (true) {
        ++cnt;

        lambda = lo + (hi-lo) / 2;
        std::tie(val_found, k_found) = this->wilber(n);

        if (k_found == k) {
            break;
        } else if (k_found < k) {
            hi = lambda;
        } else {  // k_found > k
            lo = lambda;
        }
    }
    get_actual_cost(n, kmeans_res);
    return kmeans_res;
}

std::unique_ptr<kmeans_result> kmeans_wilber::compute_interpolation_search_with_noise(size_t k, double lambda_fail){
  //const double scale = std::max(double(points.size())*1000, 1000000.0)
  //std::cout << "search with random noise added to deal with special case the easy way - k: " << k << std::endl;
  float range;
  double scale_const = 1.0;// mayeb just do sqrt of epsilon
  double eps = std::numeric_limits<double>::epsilon();
  double sqeps = sqrt(eps);
  
  // (x-eps - m)^2 = (x-eps)^2 + m^2 - 2m (x-eps) = x^2 - eps^2 + 2 eps x ... = (x-m)^2 - eps^2
  if(lambda_fail >= 1){
    range = sqeps * sqrt(lambda_fail) * scale_const;
  }
  else{
    range = sqeps * scale_const;
  }
  //range = std::min(range, 
  //std::cout << "what is range " << range << std::endl;
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd()); //Standard mersenne_twister_engine seeded with rd()
  //std::default_random_engine generator;
  //std::cout << std::setprecision(10);
  std::unique_ptr<kmeans_result> noise_res;//(new kmeans_result);
  size_t retries = 10;
  bool succ=false;
  for(size_t i=0;i < retries; ++i){
    std::uniform_real_distribution<> dis(-range, range);
    std::vector<double> new_points(points.size(), 0.0);
    for(size_t i=0; i<points.size(); ++i){
      new_points[i] = points[i] + dis(generator);    
      //std::cout << "new point and old point " << new_points[i] << " " << points[i] << std::endl;
    }
    std::sort(new_points.begin(), new_points.end());
    std::unique_ptr<kmeans_wilber> noise_wilber(new kmeans_wilber(new_points));
    std::tie(noise_res, succ) = noise_wilber->compute_interpolation_search(k, false);
    if(succ){
      //std::cout << "noise success " << std::endl;
      bestleft = std::move(noise_wilber->bestleft);
      break;      
    }
    else{
      std::cout << "noise fail - try again - scale noise range by 1.5" << std::endl;
      range = range * 1.5;
    }
  }
  assert(succ);
  return noise_res;
}


std::pair<std::unique_ptr<kmeans_result>, bool> kmeans_wilber::compute_interpolation_search(size_t k, bool add_noise_if_loop=false) {
    std::unique_ptr<kmeans_result> kmeans_res(new kmeans_result);

    double lo = 0.0;
    double lo_intercept = 0;
    size_t lo_k = n;

    double hi = is.cost_interval_l2(0, n-1);
    double hi_intercept = hi;
    size_t hi_k = 1;

    //double hi = 1e-2;

    double val_found;
    size_t k_found;
    size_t cnt = 0;
    
    while (true) {
        ++cnt;
        lambda = (hi_intercept - lo_intercept) / (lo_k - hi_k);

        std::tie(val_found, k_found) = this->wilber(n);
	//if (k_found < hi_k || k_found > lo_k) {
	//std::cout << "k_found outside range - numerical issues/empty lambda intervals " << std::endl;
	//print_vector(points);
	//assert(false);
	//}
        if (k_found <= hi_k || k_found >= lo_k) {
	  std::cout << "[Warning: K Found Outside search range - Empty Lambda Interval or numerical issues - Fix It with noise]" << std::endl;
	  std::cout << std::setprecision(20);
	  std::cout << "stats [k_found, k-range searched] " << k_found << " - (" << hi_k << " ,  " << lo_k << " ) - lambda " << lambda << std::endl;
	  // infinite loop. lambda intervals are empty between hi_k and lo_k. Add noise if allowed
	  if(add_noise_if_loop){
	    kmeans_res = compute_interpolation_search_with_noise(k, lambda);
	    k_found = k;
	  }
	  else{
	    return std::make_pair(std::move(kmeans_res), false);
	  }	  
        }
        if (k_found > k) {
            lo_k = k_found;
            lo = lambda;
            lo_intercept = val_found - lo_k * lambda;
        } else if (k_found < k) {
            hi = lambda;
            hi_k = k_found;
            hi_intercept = val_found - hi_k * lambda;
        } else {
            hi = lambda;
            break;
        }
	if(cnt > 100){
	  std::cout << "[Warning: More than 100 steps - breaking] "<< std::endl;
	  assert(false);
	}
    }   
    assert(k == k_found);
    get_actual_cost(n, kmeans_res);
    assert(kmeans_res->centers.size() == k);
    return std::make_pair(std::move(kmeans_res), true);

}

std::unique_ptr<kmeans_result> kmeans_wilber::compute(size_t k) {
    std::unique_ptr<kmeans_result> kmeans_res(new kmeans_result);
    if (k >= n) {
        kmeans_res->cost = 0.0;
        kmeans_res->centers.resize(k);
        for (size_t i = 0; i < n; ++i) {
            kmeans_res->centers[i] = points[i];
        }
        for (size_t i = n; i < k; ++i) {
            kmeans_res->centers[i] = points[n-1];
        }
        return kmeans_res;
    }
    if (k == 1) {
        kmeans_res->cost = is.cost_interval_l2(0, n-1);
        kmeans_res->centers.push_back(is.query(0, n) / ((double) n));
        return kmeans_res;
    }
    assert(std::is_sorted(points.begin(), points.end()));
    bool succ;
    switch (this->search_strat) {
    case search_strategy::BINARY:
      return compute_binary_search(k);
      break;
    case search_strategy::INTERPOLATION:
      std::tie(kmeans_res, succ) = compute_interpolation_search(k, true);
      return kmeans_res;
      break;
    default:
        throw;
    }

}

std::unique_ptr<kmeans_result> kmeans_wilber::compute_and_report(size_t k) {
    return compute(k);
}

double kmeans_wilber::weight(size_t i, size_t j) {
    if (i >= j) return std::numeric_limits<double>::max();
    return is.cost_interval_l2(i, j-1) + lambda;
}

double kmeans_wilber::g(size_t i, size_t j) {
    return f[i] + weight(i, j);
}

double kmeans_wilber::get_actual_cost(size_t n, std::unique_ptr<kmeans_result> &res) {
    double cost = 0.0;
    size_t m = n;

    std::vector<double> centers;
    std::vector<size_t> path;
    //res->path.clear();
    res->path.push_back(m);
    while (m != 0) {
        size_t prev = bestleft[m];
	res->path.push_back(prev);
        cost += is.cost_interval_l2(prev, m-1);
        double avg = is.query(prev, m) / (m - prev);
        centers.push_back(avg);
        m = prev;
    }

    res->centers.resize(centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
        res->centers[i] = centers[centers.size() - i - 1];
    }
    res->cost = cost;
    return cost;
}

std::vector<size_t> kmeans_wilber::smawk_inner(std::vector<size_t> &columns, size_t e, std::vector<size_t> &rows) {
    // base case.
    size_t n = columns.size();
    size_t result_size = (n + e - 1) / e;
    if (rows.size() == 1) {
        // result is of length (n + e - 1)/e
        return std::vector<size_t>(result_size, 0);
    }

    //reduce
    std::vector<size_t> new_rows;
    std::vector<size_t> translate;
    if (result_size < rows.size()) {
        for (size_t i = 0; i < rows.size(); ++i) {
            // I1: forall j in [0..new_rows.size() - 2]: g(new_rows[j], columns[e*j]) < g(new_rows[j+1], columns[e*j]).
            // for (size_t j = 1; j < new_rows.size(); ++j) {
            //     assert(g(new_rows[j-1], columns[e*(j-1)]) < g(new_rows[j], columns[e*(j-1)]));
            // }
            // I2: every column minima is either already in a row in new_rows OR
            //                         it is in rows[j] where j >= i.
            auto r = rows[i];
            //&& (new_rows.size() - 1 + (rows.size() - i)) >= result_size
            while (new_rows.size() &&
                   g(r, columns[e * (new_rows.size() - 1)]) <= g(new_rows.back(), columns[e * (new_rows.size() - 1)])) {
                new_rows.pop_back();
                translate.pop_back();
            }
            if (e * new_rows.size() < n) { new_rows.push_back(r); translate.push_back(i); }
        }
    } else {
        new_rows = rows;
        for (size_t i = 0; i < rows.size(); ++i) translate.push_back(i);
    }
    // assert(new_rows.size() <= result_size); // new_row.size() = ceil(n/e)
    // assert(new_rows.size());
    if (result_size == 1) {
        return std::vector<size_t>{translate[0]};
    }
    //recurse
    std::vector<size_t> column_minima_rec = smawk_inner(columns, 2*e, new_rows);  // indexes in new_rows
    // assert(column_minima_rec.size() == ((result_size + 1)/ 2));
    std::vector<size_t> column_minima; // indexes in rows

    //combine.
    column_minima.push_back(translate[column_minima_rec[0]]);
    for (size_t i = 1; i < column_minima_rec.size(); ++i) {
        size_t from = column_minima_rec[i-1];  // index in new_rows
        size_t to = column_minima_rec[i]; // index in new_rows
        size_t new_column = (2 * i - 1); // 1, 3, 5..

        // assert(column_minima.size() == new_column);

        column_minima.push_back(from);
        for (size_t r = from; r <= to; ++r) {
            if (g(new_rows[r], columns[new_column*e]) <= g(rows[column_minima[new_column]], columns[new_column*e])) {
                column_minima[new_column] = translate[r];
            }
        }
        column_minima.push_back(translate[to]);
    }
    // assert(column_minima.size() == result_size || column_minima.size() == result_size - 1);
    if (column_minima.size() < result_size) {
        // assert(column_minima.size() == result_size - 1);
        size_t from = column_minima_rec.back();
        size_t new_column = column_minima.size();

        column_minima.push_back(translate[from]);
        for (size_t r = from; r < new_rows.size(); ++r) {
            if (g(new_rows[r], columns[new_column*e]) <= g(rows[column_minima[new_column]], columns[new_column*e])) {
                column_minima[new_column] = translate[r];
            }
        }
    }
    // assert(column_minima.size() == result_size);
    return column_minima;
}

std::vector<double> kmeans_wilber::smawk(size_t i0, size_t i1, size_t j0, size_t j1, std::vector<size_t> &idxes) {
    std::vector<size_t> rows, cols;
    for (size_t i = i0; i <= i1; ++i) {
        rows.push_back(i);
    }
    for (size_t j = j0; j <= j1; ++j) {
        cols.push_back(j);
    }
    std::vector<size_t> column_minima = smawk_inner(cols, 1, rows); // indexes in rows.
    std::vector<double> res(column_minima.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = g(rows[column_minima[i]], cols[i]);
        idxes.push_back(rows[column_minima[i]]);
        assert(res[i] != std::numeric_limits<double>::max());
    }
    return res;
}
/**
 * @return f[j] = smallest value in column j0 <= j <= j1 of submatrix G[i0:i1,j0:j1] all inclusive.
 */
std::vector<double> kmeans_wilber::smawk_naive(size_t i0, size_t i1, size_t j0, size_t j1, std::vector<size_t> &idxes) {
    std::vector<double> column_minima(j1 - j0 + 1, std::numeric_limits<double>::max());
    idxes.resize(j1 - j0 + 1, n+10);
    for (size_t j = j0; j<= j1; ++j) {
        for (size_t i = i0; i <= i1; ++i) {
            if (i >= j) continue; // g(i, j) is infinity.
            double val = g(i, j);
            if (val < column_minima[j-j0]) {
                column_minima[j-j0] = val;
                idxes[j-j0] = i;
            }
        }
    }
    return column_minima;
}

std::pair<double, size_t> kmeans_wilber::wilber(size_t n) {
  //std::cout << "call " << name() << " with lambda=" << lambda << std::endl;
    f.resize(n+1, 0);
    bestleft.resize(n+1, 0);
    f[0] = 0;
    size_t c = 0, r = 0;

    while (c < n) {
        //std::cout << "hello" << std::endl;
        // step 1
        size_t p = std::min(2*c - r + 1, n);
        // step 2
        {
            std::vector<size_t> bl;
            std::vector<double> column_minima = smawk(r, c, c+1, p, bl);
            for (size_t j = c+1; j <= p; ++j) {
                f[j] = column_minima[j - ( c + 1)];
                bestleft[j] = bl[j - ( c + 1)];
            }
        }
        // step 3
        if (c+1 <= p-1) {
            std::vector<size_t> bl;
            std::vector<double> H = smawk(c + 1, p - 1, c + 2, p, bl);
            // step 4
            size_t j0 = p+1;
            for (size_t j = p; j >= c+2; --j) {
                if (H[j-(c+2)] < f[j]) j0 = j;
            }
            //step 5
            if (j0 == p+1) {
                c = p;
            } else {
                f[j0] = H[j0-(c+2)];
                bestleft[j0] = bl[j0-(c+2)];
                r = c + 1;
                c = j0;
            }
        } else {
            c = p;
        }
    }
    // find length
    size_t m = n;
    size_t length = 0;
    while (m > 0) {
        m = bestleft[m];
        ++length;
    }
    return std::make_pair(f[n], length);

}
