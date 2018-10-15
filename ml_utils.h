#ifndef ML_UTILS_H
#define ML_UTILS_H
#include <cassert>
#include <cmath>
#include <vector>
#include "utils.h"

typedef std::vector<double> featuresRow;
typedef std::vector<featuresRow> featuresSet;

double H(std::vector<double> thetas, std::vector<double> features)
{
  assert(thetas.size() == features.size());
  double sum = 0.0;
  for (int i = 0; i < features.size(); ++i)
  {
    sum += thetas[i] * features[i];
  }
  return sum;
}

double costFn(std::vector<double> thetas, featuresSet x, std::vector<double> y)
{
  double cost = 0.0;
  double sum = 0.0;
  int m = x.size();

  for (int i = 0; i < m; ++i)
  {
    sum += pow(H(thetas, x[i]) - y[i], 2);
  }
  return sum / (2 * m);
}

void thetasUpdater(std::vector<double> &thetas, double alpha, const featuresSet &x, const std::vector<double> &y) {
  std::vector<double> tmpThetas;
  double tmpSum;
  for(int t = 0; t < thetas.size(); ++t) {
    tmpSum = 0.0;
    for(int i = 0; i < x[t].size(); ++i) {
      tmpSum += (H(thetas, x[i]) - y[i]) * x[t][i];
    }
    tmpThetas.push_back(thetas[t] - ((alpha / (2 * x[t].size())) * tmpSum));
  }
  
  thetas = tmpThetas;
}

#endif

