#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <Eigen/Dense>
#include <vector>

class Perceptron {
private:
  double w0, w1, w2;

  int StepFunction(double z) const;
  double WeightedSumOfInputs(double x1, double x2) const;

public:
  Perceptron(double w0Val = 0.0, double w1Val = 0.0, double w2Val = 0.0);

  int Predict(double x1, double x2) const;
  double PredictBoundary(double input) const;
  void Fit(const Eigen::MatrixXd &data, const Eigen::MatrixXd &label, int epochs, double lr);
};

#endif
