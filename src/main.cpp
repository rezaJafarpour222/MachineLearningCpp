#include "matrixBuilder.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
using namespace std;
int main() {
  int mean = 0;
  int number = 1000;
  double stdev = 0.1;
  double diff = 0.35;
  Eigen::MatrixXd ds1 = buildMatrix(42, mean, stdev, number, 0);
  Eigen::MatrixXd ds2 = buildMatrix(123, mean, stdev, number, 1, diff);
  Eigen::MatrixXd fullShuffledDs = MergeAndShuffle(ds1, ds2, 1000);
  cout << "Shuffled data set" << fullShuffledDs.topRows(20);
}