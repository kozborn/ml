#ifndef MATRIX_ML_UTILS_H
#define MATRIX_ML_UTILS_H

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

double logisticCostFn(const featuresSet &x, const std::vector<double> &y, const std::vector<double> &thetas)
{
  double cost = 0.0;
  double g = 0.0;
  double l = 0.0;
  double lg = 0.0;
  int m = x.size();
  for (int i = 0; i < m; ++i)
  {
    g = G(H(x[i], thetas));
    l = log(g);
    lg = log(1 - g);
    cost += y[i] * l + (1 - y[i]) * lg;
  }

  return -1 * cost / m;
}

std::vector<double> logisticRegressionCosts(const featuresSet &x, const std::vector<double> &y, const std::vector<double> &thetas)
{
  int m = x.size();
  std::vector<double> costs;
  for (int i = 0; i < m; ++i)
  {
    costs.push_back(logisticCostFn(x, y, thetas) - y[i]);
  }
  return costs;
  return costs;
}

#endif