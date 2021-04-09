#include <iomanip>
#include <iostream>
#include <tuple>
#include <typeinfo>
#include <sstream>
#include <cassert>
#include <algorithm>
#include "kmeans.h"

using namespace std;


template <typename T>
void print_vector(const std::vector<T> & vec, std::string sep=", "){
    for(auto elem : vec)
    {
        std::cout << elem << sep;
    }
    std::cout<<std::endl;
}

template<typename T>
bool check_equal(const std::vector<T> & myvector){
  return std::adjacent_find( myvector.begin(), myvector.end(), std::not_equal_to<T>() ) == myvector.end();
}


std::vector<double> uniform_sample(size_t n, double low=-1.0, double high=1.0){ 
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(low, high);
  std::vector<double> new_points(n, 0.0);
  //std::cout << std::setprecision(10);
  for(size_t i=0;i < n; ++i){
    new_points[i] = dis(generator);
  }
  std::sort(new_points.begin(), new_points.end());
  return new_points;
}

double get_cost(kmeans &km, size_t k) {
    std::unique_ptr<kmeans_result> res(km.compute_and_report(k));
    return res->cost;
}

double get_cost(std::unique_ptr<kmeans_result> &res, std::vector<double> &points, size_t k) {
    double computed_cost = 0;
    for (auto point : points) {
        double closest = res->centers[0];
        for (auto center : res->centers) {
            if (abs(point - center) < abs(point - closest)) closest = center;
        }
        computed_cost += (point-closest) * (point - closest);
    }
    return computed_cost;
}


bool check_clustering_size(std::unique_ptr<kmeans_result> &res, std::vector<double> &points, size_t k) {
    if (res->centers.size() != k || res->centers.size() < 1) return false;
    return true;
}

bool check_clustering_cost(std::unique_ptr<kmeans_result> &res, std::vector<double> &points, size_t k) {
    double computed_cost = get_cost(res, points, k);
    if (abs(computed_cost - res->cost) > 1e-6) return false;
    return true;
}


void test_cluster_cost_equal_returned_cost_no_lambda_diff() {
    std::string prefix_template = "[test_cluster_cost_equal_returned_cost_no_lambda_diff] ";
    std::vector<double> points = {
        1, 2, 3,
        101, 102, 103,
        201, 202, 203,
    };

    std::unique_ptr<kmeans_linear> linear(new kmeans_linear(points));
    std::unique_ptr<kmeans_slow> slow(new kmeans_slow(points));
    std::unique_ptr<kmeans_monotone> monotone(new kmeans_monotone(points));
    std::unique_ptr<kmeans> wilber(new kmeans_wilber(points));
    
    bool fail = false;
    for (size_t k = 1; k < 10; ++k) {
      std::vector<size_t> sizes;
      std::vector<double> scores;

      std::cout << "[k = " << k << "] Testing" << std::endl;
      std::unique_ptr<kmeans_result> linear_res = linear->compute_and_report(k);
      //std::cout << "linear " << std::endl;
      //print_vector(linear->row);
      std::unique_ptr<kmeans_result> monotone_res = monotone->compute_and_report(k);
      //std::cout << "monotone " << std::endl;
      //print_vector(monotone->row);      
      std::unique_ptr<kmeans_result> slow_res = slow->compute_and_report(k);
      //std::cout << "slow " << std::endl;
      //print_vector(slow->row);            
      std::unique_ptr<kmeans_result> wilber_res = wilber->compute_and_report(k);
      
        std::stringstream ss;
        ss << prefix_template << "[ k = " << k << " ] ";
        std::string prefix = ss.str();
	sizes.push_back(linear_res->centers.size());
	scores.push_back(linear_res->cost);
	sizes.push_back(monotone_res->centers.size());
	scores.push_back(monotone_res->cost);
	sizes.push_back(slow_res->centers.size());
	scores.push_back(slow_res->cost);
	sizes.push_back(wilber_res->centers.size());
	scores.push_back(wilber_res->cost);
	if (!check_equal(scores)){
	  std::cout << "scores not equal " << std::endl;
	  for (auto i: scores)
	    std::cout << i << ", ";
	  std::cout << std::endl;
	}
	if (!check_equal(sizes)){
	  std::cout << "center sizes not equal " << std::endl;
	  for (auto i: sizes)
	    std::cout << i << ", ";
	  std::cout << std::endl;
	}
			 
        if (!check_clustering_size(linear_res, points, k)) {
            cout << prefix << "linear clustering expected size " << k
                 << " found size " << linear_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(linear_res, points, k)) {
            cout << prefix << "linear clustering cost failed. Returned cost " << linear_res->cost
                 << "  Computed cost " << get_cost(linear_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(monotone_res, points, k)) {
            cout << prefix << "monotone clustering expected size " << k
                 << " found size " << monotone_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(monotone_res, points, k)) {
            cout << prefix << "monotone clustering cost failed. Returned cost " << monotone_res->cost
                 << "  Computed cost " << get_cost(monotone_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(slow_res, points, k)) {
            cout << prefix << "slow clustering expected size " << k
                 << " found size " << slow_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(slow_res, points, k)) {
            cout << prefix << "slow clustering cost failed. Returned cost " << slow_res->cost
                 << "  Computed cost " << get_cost(slow_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(wilber_res, points, k)) {
            cout << prefix << "wilber clustering expected size " << k
                 << " found size " << wilber_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(wilber_res, points, k)) {
            cout << prefix << "wilber clustering cost failed. Returned cost " << wilber_res->cost
                 << "  Computed cost " << get_cost(wilber_res, points, k) << endl;
            fail = true;
        }
	
    }
    
    if (fail) {
        cout << prefix_template << "Failed." << endl;
    } else {
        cout << prefix_template << "Succeeded." << endl;
    }
}


void correctness_test_no_lambda_difference() {

    std::vector<double> points = {
        1, 2, 3,
        101, 102, 103,
        201, 202, 203,
    };
    std::unique_ptr<kmeans> wilber(new kmeans_wilber(points));
    for(size_t k=1; k< 10;++k){
      //size_t k = 5;
      double res = get_cost(*wilber, k);    
      std::cout <<  "[k = " << k << "] correctness no lambda difference works - cost found" << " " << res << std::endl;
    }
    //std::unique_ptr<kmeans_result> res(*wilber.compute_and_report(k));
    
}

void correctness_test_random() {
    std::vector<double> points = {0.0041841036041334601, 0.016864905913439476,
                                  0.091539430201843741, 0.11167850389725253,
                                  0.11729255208759837, 0.15870772838060987,
                                  0.21537383129510801, 0.22030075252311732,
                                  0.29234574608234609, 0.34182095515978905,
                                  0.38064794144662972, 0.42369328807073692,
                                  0.42898263636024347, 0.46299304217492687,
                                  0.59849854723755469, 0.77144917504818644,
                                  0.78318033400636167, 0.8393332644552387,
                                  0.92763049366511063, 0.98685245969033264};

    std::unique_ptr<kmeans> linear(new kmeans_linear(points));
    std::unique_ptr<kmeans> slow(new kmeans_slow(points));
    std::unique_ptr<kmeans> monotone(new kmeans_monotone(points));
    std::unique_ptr<kmeans> wilber(new kmeans_wilber(points));

    cout << "Running tests for linear, slow and monotone algorithms." << endl;
    bool any_fail = false;
    for (size_t k = 1; k < 10; ++k) {
        double linear_res = get_cost(*linear, k);
        double slow_res = get_cost(*slow, k);
        double monotone_res = get_cost(*monotone, k);
        double wilber_res = get_cost(*wilber, k);
        bool fail = false;
        if (linear_res != slow_res) {
            cout << "[k = " << k << "] Test failed for linear" << endl;
            fail = true;
        } else {
            cout << "[k = " << k << "] Test succeeded for slow" << endl;
        }
        if (slow_res != monotone_res) {
            cout << "[k = " << k << "] Test failed for monotone" << endl;
            fail = true;
        } else {
            cout << "[k = " << k << "] Test succeeded for monotone" << endl;
        }
        if (slow_res != wilber_res) {
            cout << "[k = " << k << "] Test failed for wilber" << endl;
            fail = true;
        } else {
            cout << "[k = " << k << "] Test succeeded for wilber" << endl;
        }
        if (fail) {
            any_fail = true;
            double diff_linear_slow = abs(linear_res - slow_res);
            double diff_monotone_slow = abs(monotone_res - slow_res);
            double diff_wilber_slow = abs(wilber_res - slow_res);

            cout << "[k = " << k << "] Failed" << endl;
            cout << "Additional info below:" << endl;
            cout << "[k = " << k << "] ";
            cout << "|linear - slow| = " << setprecision(10) << diff_linear_slow << endl;

            cout << "[k = " << k << "] ";
            cout << "|monotone - slow| = " << setprecision(10) << diff_monotone_slow << endl;

            cout << "[k = " << k << "] ";
            cout << "|wilber - slow| = " << setprecision(10) << diff_wilber_slow << endl;
        }
    }
    if (any_fail) {
        cout << "[correctness_test_random] Failed" << endl;
    } else {
        cout << "[correctness_test_random] Succeeded" << endl;
    }
}

void correctness_lloyd() {
    cout << "Running tests for Lloyds algorithm" << endl;
    bool fail = false;
    std::vector<double> points = {0.0041841036041334601, 0.016864905913439476,
                                  0.091539430201843741, 0.11167850389725253,
                                  0.11729255208759837, 0.15870772838060987,
                                  0.21537383129510801, 0.22030075252311732,
                                  0.29234574608234609, 0.34182095515978905,
                                  0.38064794144662972, 0.42369328807073692,
                                  0.42898263636024347, 0.46299304217492687,
                                  0.59849854723755469, 0.77144917504818644,
                                  0.78318033400636167, 0.8393332644552387,
                                  0.92763049366511063, 0.98685245969033264};

    std::unique_ptr<kmeans_lloyd> lloyd_slow(new kmeans_lloyd_slow(points));
    std::unique_ptr<kmeans_lloyd> lloyd_fast(new kmeans_lloyd_fast(points));

    for (size_t seed = 13; seed < 13*13; seed += 13) {
        for (size_t k = 1; k <= 10; ++k) {
            lloyd_fast->set_seed(seed);
            double cost_fast = get_cost(*lloyd_fast, k);
            lloyd_slow->set_seed(seed);
            double cost_slow = get_cost(*lloyd_slow, k);
            if (cost_fast < 0) {
                fail = true;
                cout << "[lloyd] Error [fast], negative cost: " << cost_fast << std::endl;
            }
            if (cost_slow < 0) {
                cout << "[lloyd] Error [slow], negative cost: " << cost_slow << std::endl;
            }
            if (cost_fast != cost_slow) {
                cout << "Lloyd Error for k = " << k << "  seed = " << seed << endl;
                cout << "  cost_slow: " << cost_slow << "   cost: " << cost_fast << endl;
                fail = true;
            }
        }
    }
    if (fail) {
        cout << "[correctness_lloyd] Failed." << endl;
    } else {
        cout << "[correctness_lloyd] Succeeded." << endl;
    }
}


void test_cluster_cost_equal_returned_cost() {
    std::string prefix_template = "[test_cluster_cost_equal_returned_cost] ";
    std::vector<double> points = {0.0041841036041334601, 0.016864905913439476,
                                  0.091539430201843741, 0.11167850389725253,
                                  0.11729255208759837, 0.15870772838060987,
                                  0.21537383129510801, 0.22030075252311732,
                                  0.29234574608234609, 0.34182095515978905,
                                  0.38064794144662972, 0.42369328807073692,
                                  0.42898263636024347, 0.46299304217492687,
                                  0.59849854723755469, 0.77144917504818644,
                                  0.78318033400636167, 0.8393332644552387,
                                  0.92763049366511063, 0.98685245969033264};
    std::unique_ptr<kmeans> linear(new kmeans_linear(points));
    std::unique_ptr<kmeans> slow(new kmeans_slow(points));
    std::unique_ptr<kmeans> monotone(new kmeans_monotone(points));
    std::unique_ptr<kmeans> wilber(new kmeans_wilber(points));
    bool fail = false;
    for (size_t k = 1; k < 10; ++k) {
      std::cout << "testing " << k << std::endl;
        std::unique_ptr<kmeans_result> linear_res = linear->compute_and_report(k);
        std::unique_ptr<kmeans_result> monotone_res = monotone->compute_and_report(k);
        std::unique_ptr<kmeans_result> slow_res = slow->compute_and_report(k);
        std::unique_ptr<kmeans_result> wilber_res = wilber->compute_and_report(k);
        std::stringstream ss;
        ss << prefix_template << "[ k = " << k << " ] ";
        std::string prefix = ss.str();
        if (!check_clustering_size(linear_res, points, k)) {
            cout << prefix << "linear clustering expected size " << k
                 << " found size " << linear_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(linear_res, points, k)) {
            cout << prefix << "linear clustering cost failed. Returned cost " << linear_res->cost
                 << "  Computed cost " << get_cost(linear_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(monotone_res, points, k)) {
            cout << prefix << "monotone clustering expected size " << k
                 << " found size " << monotone_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(monotone_res, points, k)) {
            cout << prefix << "monotone clustering cost failed. Returned cost " << monotone_res->cost
                 << "  Computed cost " << get_cost(monotone_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(slow_res, points, k)) {
            cout << prefix << "slow clustering expected size " << k
                 << " found size " << slow_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(slow_res, points, k)) {
            cout << prefix << "slow clustering cost failed. Returned cost " << slow_res->cost
                 << "  Computed cost " << get_cost(slow_res, points, k) << endl;
            fail = true;
        }
        if (!check_clustering_size(wilber_res, points, k)) {
            cout << prefix << "wilber clustering expected size " << k
                 << " found size " << wilber_res->centers.size() << "." << endl;
            fail = true;
        } else if (!check_clustering_cost(wilber_res, points, k)) {
            cout << prefix << "wilber clustering cost failed. Returned cost " << wilber_res->cost
                 << "  Computed cost " << get_cost(wilber_res, points, k) << endl;
            fail = true;
        }
	
    }
    if (fail) {
        cout << prefix_template << "Failed." << endl;
    } else {
        cout << prefix_template << "Succeeded." << endl;
    }
}

bool run_test(std::vector<double> & points, int k){
    std::unique_ptr<kmeans_dp> linear(new kmeans_linear(points));
    std::unique_ptr<kmeans_dp> slow(new kmeans_slow(points));
    std::unique_ptr<kmeans_dp> monotone(new kmeans_monotone(points));
    std::unique_ptr<kmeans> wilber(new kmeans_wilber(points));
    
    bool fail = false;
    //std::cout << "Testing iteration " << i << " data points " << k << " clusters " << std::endl;
    //print_vector(points);
    std::vector<size_t> sizes;
    std::vector<double> scores;
    std::cout << "\r" << "[k = " << k <<  ", data size = " << points.size() << "] Testing" << std::flush;
    std::unique_ptr<kmeans_result> linear_res = linear->compute_and_report(k);
    std::unique_ptr<kmeans_result> monotone_res = monotone->compute_and_report(k);
    std::unique_ptr<kmeans_result> slow_res = slow->compute_and_report(k);
    std::unique_ptr<kmeans_result> wilber_res = wilber->compute_and_report(k);
    sizes.push_back(linear_res->centers.size());
    scores.push_back(linear_res->cost);
    sizes.push_back(monotone_res->centers.size());
    scores.push_back(monotone_res->cost);
    sizes.push_back(slow_res->centers.size());
    scores.push_back(slow_res->cost);
    sizes.push_back(wilber_res->centers.size());
    scores.push_back(wilber_res->cost);
      
    if (!check_equal(sizes)){
      std::cout << "center sizes not equal " << std::endl;
      for (auto i: sizes)
	std::cout << i << ", ";	
      std::cout << std::endl << "linear final row" << std::endl;	
      print_vector(linear->row);
      std::cout << "slow final row" << std::endl;	
      print_vector(slow->row);
      std::cout << "monotone final row" << std::endl;	
      print_vector(monotone->row);	
      std::cout << std::endl;
      return false;
    }
    if (!check_equal(scores)){
      const auto [smin, smax] = std::minmax_element(scores.begin(), scores.end());
      double diff = *smax - *smin;
      if (diff > 0.0005){	  
	std::cout << "scores not equal " << diff << " " << *smin << " " << *smax << std::endl;
	std::cout << std::setprecision(10);
	print_vector(scores);
	std::cout << std::endl;
	return false;
	//assert(false);
      }
    }
    return true;
}

  
void run_random_tests(size_t k, size_t repeats){
  std::string prefix_template = "[Random Tests] ";
  bool succ=true;
  for(size_t i=k; i < 4 * k; i++){
    for (size_t r=0; r < repeats; ++r){
      std::vector<double> points = uniform_sample(i, -100.0, 100.0);
      for(size_t j=1; j< i;++j){
	succ = run_test(points, j);
	if(!succ){
	  std::cout << prefix_template << " FAILED " << std::endl;
	  assert(false);
	  return;
	}
      }
    }    
  }
  std::cout << prefix_template << " SUCCEDED " << std::endl;    
}


void more_clusters_than_points() {
    // TODO: test it.
    std::vector<double> points = {1.0, 2.0, 3.0, 4.0};
    size_t k = 10;
    std::vector<std::shared_ptr<kmeans> > algs = {
        std::shared_ptr<kmeans>(new kmeans_linear(points)),
        std::shared_ptr<kmeans>(new kmeans_slow(points)),
        std::shared_ptr<kmeans>(new kmeans_monotone(points)),
        std::shared_ptr<kmeans>(new kmeans_wilber(points)),
        std::shared_ptr<kmeans>(new kmeans_lloyd_slow(points)),
        std::shared_ptr<kmeans>(new kmeans_lloyd_fast(points))
    };
    bool fail = false;
    for (size_t i = 0; i < algs.size(); ++i) {
        double cost = get_cost(*algs[i], k);
        if (cost >= 1e-6) {
            std::cout << "[more_clusters_than_points] " << typeid(algs[i]).name() << " was not 0." << std::endl;
            fail = true;
        }
    }
    if (fail) {
        std::cout << "[more_clusters_than_points] Failed" << std::endl;
        return;
    }
    std::cout << "[more_clusters_than_points] Succeeded" << std::endl;
}


void empty_interval_tests() {
    std::string prefix_template = "[Empty Lambda Interval Tests]";
    bool succ;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 generator(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.2, 2);

    for(size_t cluster_size=3; cluster_size < 20; ++cluster_size){
      for(size_t num_clusters=1; num_clusters<10;++num_clusters){
	std::vector<double> points;
	double random_dist =  dis(generator);
	for(size_t i=0; i < cluster_size; ++i){	  
	  double tmp = (double) i * random_dist;
	  for(size_t j=0; j < num_clusters; ++j){
	    points.push_back(tmp + j * 500);
	  }
	}      	
	std::sort(points.begin(), points.end());
	for(size_t k=1;k< points.size();++k){
	  succ = run_test(points, k);
	  if(!succ){
	    print_vector(points);
	    assert(false);
	  }
	}
      }
    }
    std::cout << prefix_template << " Succeeded" << std::endl;    
}



int main(int argc, char *argv[]) {
    if (argc == 1 || argv[1] == std::string("all")) {
      //correctness_test_random();
      //correctness_lloyd();
      //more_clusters_than_points();
      //test_cluster_cost_equal_returned_cost();
      empty_interval_tests();
      for(size_t i=0;i < 1;++i){
	std::cout << "\r" << "Lambda No diff [i = " << i << "] " << std::flush;
	test_cluster_cost_equal_returned_cost_no_lambda_diff();
      }
      correctness_test_no_lambda_difference();
      run_random_tests(3, 100);
      return 0;
    }
    if (argc == 1) {
        return 1;
    }
    return 0;
}
