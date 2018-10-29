#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <cassert>
#include <cmath>
#include "utils.h"
#include "ml_utils.h"

double logisticH(const std::vector<double> &features, const std::vector<double> &thetas)
{
  assert(thetas.size() == features.size());
  double prediction = 0.0;
  for (int j = 0; j < features.size(); ++j)
  {
    prediction += thetas[j] * features[j];
  }
  return prediction;
}

double G(double HResult)
{
  return 1 / (1 + std::exp(-HResult));
}

double logisticCostFn(const featuresSet &x, const resultsSet &y, const std::vector<double> &thetas)
{
  double cost = 0.0;
  double g = 0.0;
  double l = 0.0;
  double lg = 0.0;
  int m = x.size();
  for (int i = 0; i < m; ++i)
  {
    g = G(logisticH(x[i], thetas));
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
}

std::vector<double> gradientDescentLogisticVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, const double alpha)
{
  double firstCost = 0.0;
  double initialCost = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;

  std::vector<double> t = thetas;
  firstCost = logisticCostFn(X, Y, t);
  std::vector<double> costs;
  do
  {
    initialCost = logisticCostFn(X, Y, t);

    costs = logisticRegressionCosts(X, Y, t);
    thetasUpdater(X, Y, t, alpha, costs);
    costAfter = logisticCostFn(X, Y, t);

    iterationCount++;
    if (isEqual(costAfter, initialCost) || isEqual(costAfter, 0))
      break;
  } while (costAfter < initialCost);
  std::cout << "Initial cost: " << firstCost << " cost after: " << costAfter << std::endl;
  std::cout << "Iteration count: " << iterationCount << std::endl;
  return t;
}

void logisticRegressionCodeIterative(const featuresSet &x, const resultsSet &y, const double alpha)
{
  std::cout << "Logistic version of gradient descent" << std::endl;
  int iterationCount = 0;
  std::vector<double> costs;
  std::vector<double> thetas(x[0].size(), 1);

  std::cout << "X matrix" << std::endl;
  for (int i = 0; i < x.size(); ++i)
  {
    print(x[i]);
  }

  std::cout << std::endl;
  std::cout << "Y vector" << std::endl;
  print(y);

  std::cout << "Initial thetas" << std::endl;
  print(thetas);

  std::vector<double> calculatedThetas;
  calculatedThetas = gradientDescentLogisticVersion(x, y, thetas, alpha);
  std::cout << "Calculated thetas" << std::endl;
  print(calculatedThetas);
  std::cout << std::endl
            << std::endl;

  std::vector<double> testX1 = {1, 2};
  std::vector<double> testX2 = {1, 45};

  std::cout << "Test values: " << std::endl;
  std::cout << "testX1: " << G(logisticH(testX1, calculatedThetas)) << std::endl;
  std::cout << "testX2: " << G(logisticH(testX2, calculatedThetas)) << std::endl;
}

#endif