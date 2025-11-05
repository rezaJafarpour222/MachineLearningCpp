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
  int Preditct(double x1, double x2) {
    double z = WeightedSumOfInputs(x1, x2);
    return StepFunction(z);
  }
  double PredictBoundary(double x) {
    double sum = -(w1 * x + w0) / w2;
    return sum;
  }
  double Fit(vector<double> x, vector<double> y, int epochs = 1, double lr = 0.005) {
    vector<int> errors;
    for (int i = 0; i < epochs; i++) {
      int error = 0;
      for (int value : x) {
      }
    }
  }
};