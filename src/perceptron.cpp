#include <Eigen/Dense>
#include <vector>
using namespace std;
class Perceptron {

  double w0, w1, w2;

public:
  Perceptron(double w0Val, double w1Val, double w2Val) : w0(w0Val), w1(w1Val), w2(w2Val) {
  }

  int StepFunction(double z) {
    return z >= 0 ? 1 : 0;
  }

  double WeightedSumOfInputs(double x1, double x2) const {
    double sum = (1 * w0) + (x1 * w1) + (x2 * w2);
    return sum;
  }
  int Predict(double x1, double x2) {
    double z = WeightedSumOfInputs(x1, x2);
    return StepFunction(z);
  }
  double PredictBoundary(double input) {
    double sum = -(w1 * input + w0) / w2;
    return sum;
  }
  void Fit(Eigen::MatrixXd &data, Eigen::MatrixXd &label, int epochs = 1, double lr = 0.005) {
    vector<int> errors;
    for (int epoch = 0; epoch < epochs; epoch++) {
      int error = 0;
      for (int i = 0; i < data.rows(); i++) {
        double x = data(i, 0);
        double y = data(i, 1);
        double category = label(i);
        double update = lr * (category - Predict(x, y));
        w0 += update;
        w1 += update * x;
        w2 += update * y;
        if (update != 0)
          error++;
      }
      errors.push_back(error);
    }
  }
};