#ifndef ML_UTILS_H
#define ML_UTILS_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "utils.h"

typedef std::vector<double> resultsSet;
typedef std::vector<double> featuresRow;
typedef std::vector<featuresRow> featuresSet;

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

void thetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha)
{
  std::vector<double> tmpThetas;

  std::vector<double> costs;
  int m = x.size();
  int n = thetas.size();
  double prediction = 0.0;
  costs = linearRegressionCosts(x, y, thetas);
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
  for (int i = 0; i < x.size(); ++i)
  {
    double avg = (max[i] - min[i]) / 2;
    for (int j = 1; j < x[i].size(); ++j)
    {
      x[i][j] = (x[i][j] - avg) / (max[i] - min[i]);
    }
  }
}

#endif
