#ifndef ML_UTILS_H
#define ML_UTILS_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "utils.h"

double H(const std::vector<double> &features, const std::vector<double> &thetas)
{
  assert(thetas.size() == features.size());
  double prediction = 0.0;
  for (int j = 0; j < features.size(); ++j)
  {
    prediction += thetas[j] * features[j];
  }
  return prediction;
}

double costFn(featuresSet x, std::vector<double> y, std::vector<double> thetas)
{
  double sum = 0.0;
  int m = x.size();

  for (int i = 0; i < m; ++i)
  {
    sum += pow(H(x[i], thetas) - y[i], 2);
  }
  return sum / (2 * m);
}

std::vector<double> linearRegressionCosts(const featuresSet &x, const std::vector<double> &y, const std::vector<double> &thetas)
{
  int m = x.size();
  std::vector<double> costs;
  for (int i = 0; i < m; ++i)
  {
    costs.push_back(H(x[i], thetas) - y[i]);
  }
  return costs;
}

void thetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha, const std::vector<double> &costs)
{
  std::vector<double> tmpThetas;

  int m = x.size();
  int n = thetas.size();
  double prediction = 0.0;
  for (int j = 0; j < n; ++j)
  {
    prediction = 0.0;
    // INSIDE SUM i=0 to m
    for (int i = 0; i < m; ++i)
    {
      prediction += costs[i] * x[i][j];
    }
    tmpThetas.push_back(thetas[j] - ((alpha / m) * prediction));
  }
  thetas = tmpThetas;
}

void scaleFeatures(featuresSet &x, std::vector<double> min, std::vector<double> max)
{
  double avg = 0.0;
  double delta = 1;
  for (int j = 0; j < x[0].size(); ++j)
  {
    delta = max[j] - min[j];
    avg = (delta) / 2;
    for (int i = 0; i < x.size(); ++i)
    {
      x[i][j] = (x[i][j] - avg) / (delta);
    }
  }
}

#endif
