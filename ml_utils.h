#ifndef ML_UTILS_H
#define ML_UTILS_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "utils.h"
#include <Eigen/Dense>

typedef std::vector<double> resultsSet;
typedef std::vector<double> featuresRow;
typedef std::vector<featuresRow> featuresSet;

// Theta updater from vectorization = theta = theta - alpha/m * (X' * (X*theta - Y))
Eigen::MatrixXd normalEquation(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
  return (X.transpose() * X).completeOrthogonalDecomposition().pseudoInverse() * X.transpose() * Y;
}

double G(double HResult)
{
  return 1 / (1 + std::exp(-HResult));
}

double H(std::vector<double> thetas, std::vector<double> features)
{
  assert(thetas.size() == features.size());
  double prediction = 0.0;
  for (int j = 0; j < features.size(); ++j)
  {
    prediction += thetas[j] * features[j];
  }
  return prediction;
}

double logisticCostFn(const std::vector<double> &thetas, const featuresSet &x, const std::vector<int> &y)
{
  double cost = 0.0;
  double g = 0.0;
  double l = 0.0;
  double lg = 0.0;
  int m = x.size();
  for (int i = 0; i < m; ++i)
  {
    g = G(H(thetas, x[i]));
    l = log(g);
    lg = log(1 - g);
    cost += y[i] * l + (1 - y[i]) * lg;
    std::cout << cost << std::endl;
  }

  return -1 * cost / m;
}

double costFn(std::vector<double> thetas, featuresSet x, std::vector<double> y)
{
  double sum = 0.0;
  int m = x.size();

  for (int i = 0; i < m; ++i)
  {
    sum += pow(H(thetas, x[i]) - y[i], 2);
  }
  return sum / (2 * m);
}

std::vector<double> linearRegressionCosts(const featuresSet &x, const std::vector<double> &y, const std::vector<double> &thetas)
{
  int m = x.size();
  std::vector<double> costs;
  for (int i = 0; i < m; ++i)
  {
    costs.push_back(H(thetas, x[i]) - y[i]);
  }
  return costs;
}

void thetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha, const std::vector<double> &costs)
{
  std::vector<double> tmpThetas;
  double tmpSum;
  int m = x.size();
  int n = thetas.size();
  double prediction = 0.0;

  for (int j = 0; j < n; ++j)
  {
    prediction = 0.0;
    // INSIDE SUM i=0 to m
    for (int i = 0; i < m; ++i)
    {
      prediction += (costs[i]) * x[i][j];
    }
    tmpThetas.push_back(thetas[j] - ((alpha / m) * prediction));
  }
  thetas = tmpThetas;
}

void scaleFeatures(featuresSet &x, double min, double max)
{
  double avg = (max - min) / 2;
  for (int i = 0; i < x.size(); ++i)
  {
    for (int j = 0; j < x[i].size(); ++j)
    {
      x[i][j] = (x[i][j] - avg) / (max - min);
    }
  }
}

#endif
