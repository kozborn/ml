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

void thetasUpdater(std::vector<double> &thetas, double alpha, const featuresSet &x, const std::vector<double> &y)
{
  std::vector<double> tmpThetas;
  double tmpSum;
  int m = x.size();
  int n = thetas.size();
  double prediction = 0.0;
  std::vector<double> diffs;

  for (int i = 0; i < m; ++i)
  {
    diffs.push_back(H(thetas, x[i]) - y[i]);
  }

  for (int j = 0; j < n; ++j)
  {
    prediction = 0.0;
    // INSIDE SUM i=0 to m
    for (int i = 0; i < m; ++i)
    {
      prediction += (diffs[i]) * x[i][j];
    }
    tmpThetas.push_back(thetas[j] - ((alpha / m * prediction)));
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
