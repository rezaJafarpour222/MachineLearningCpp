#include "perceptron.h"
using namespace std;

int Perceptron::StepFunction(double z) const {
  return z >= 0 ? 1 : 0;
}

double Perceptron::WeightedSumOfInputs(double x1, double x2) const {
  return w0 + w1 * x1 + w2 * x2;
}

Perceptron::Perceptron(double w0Val, double w1Val, double w2Val) : w0(w0Val), w1(w1Val), w2(w2Val) {
}

int Perceptron::Predict(double x1, double x2) const {
  double z = WeightedSumOfInputs(x1, x2);
  return StepFunction(z);
}

double Perceptron::PredictBoundary(double input) const {
  return -(w1 * input + w0) / w2;
}

void Perceptron::Fit(const Eigen::MatrixXd &fullData, int epochs, double lr) {
  Eigen::MatrixXd data = fullData.leftCols(fullData.cols() - 1);
  Eigen::MatrixXd label = fullData.rightCols(1);
  vector<int> errors;
  for (int epoch = 0; epoch < epochs; epoch++) {
    int error = 0;
    for (int i = 0; i < data.rows(); i++) {
      double x = data(i, 0);
      double y = data(i, 1);
      double category = label(i, 0);
      double update = lr * (category - Predict(x, y));

      w0 += update;
      w1 += update * x;
      w2 += update * y;
      if (update != 0)
        error++;
    }
    errors.push_back(error);
  }
};