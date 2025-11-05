#ifndef MATRIXBUILDER_H
#define MATRIXBUILDER_H

#include <Eigen/Dense>
#include <random>
using namespace std;
Eigen::MatrixXd buildMatrix(int seed, double mean, double stdev, int n, int set_type = 0, double diff = 0) {
  mt19937 gen(42); // Seed
  normal_distribution<double> dist_x1(mean, stdev);
  normal_distribution<double> dist_x2(mean, stdev);
  Eigen::VectorXd x1(n), x2(n), type(n);

  if (set_type == 0) {

    type = Eigen::VectorXd::Zero(n);
  } else if (set_type == 1) {
    type = Eigen::VectorXd::Ones(n);
  } else {
    throw runtime_error("only two category right now");
  }

  for (int i = 0; i < n; i++) {
    x1(i) = dist_x1(gen) + diff;
    x2(i) = dist_x2(gen) + diff;
  }
  Eigen::MatrixXd ds(n, 3);
  ds.col(0) = x1;
  ds.col(1) = x2;
  ds.col(2) = type;
  return ds;
}
Eigen::MatrixXd MergeAndShuffle(Eigen::MatrixXd &ds1, Eigen::MatrixXd &ds2, int n) {
  Eigen::MatrixXd fullDs(2 * n, 3);
  fullDs.topRows(n) = ds1;
  fullDs.bottomRows(n) = ds2;

  std::mt19937 gen(2025);
  std::vector<int> indices(2 * n);

  for (int i = 0; i < 2 * n; i++)
    indices[i] = i;

  std::shuffle(indices.begin(), indices.end(), gen);

  Eigen::MatrixXd shuffledDs(2 * n, 3);
  for (int i = 0; i < 2 * n; i++) {
    shuffledDs.row(i) = fullDs.row(indices[i]);
  }
  return shuffledDs;
}
#endif