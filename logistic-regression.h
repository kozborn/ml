#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <cassert>
#include <cmath>
#include <chrono>
#include "utils.h"
#include "ml_utils.h"

double logisticH(const std::vector<double> &features, const std::vector<double> &thetas)
{
  // std::cout << "Thetas.size() = " << thetas.size() << " features.size() = " << features.size() << std::endl;
  assert(thetas.size() == features.size());
  double prediction = 0.0;
  for (int j = 0; j < features.size(); ++j)
  {
    prediction += thetas[j] * features[j];
  }
  return 1.0 / (1.0 + std::exp(-prediction));
}

double logisticCostFn(const featuresSet &x, const resultsSet &y, const std::vector<double> &costs)
{
  double cost = 0.0;
  double g = 0.0;
  int m = x.size();
  for (int i = 0; i < m; ++i)
  {
    g = costs[i];
    cost += y[i] * std::log(g) + (1 - y[i]) * std::log(1 - g);
  }

  return -1 * cost / m;
}

std::vector<double> logisticRegressionCosts(const featuresSet &x, const std::vector<double> &thetas)
{
  std::vector<double> costs;
  for (int i = 0; i < x.size(); ++i)
  {
    costs.push_back(logisticH(x[i], thetas));
  }
  return costs;
}

void logisticThetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha, const std::vector<double> &costs)
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
      prediction += (costs[i] - y[i]) * x[i][j];
    }
    tmpThetas.push_back(thetas[j] - ((alpha / m) * prediction));
  }
  thetas = tmpThetas;
}

std::vector<double> gradientDescentLogisticVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, double alpha)
{
  double costBefore = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;

  std::vector<double> t = thetas;
  std::vector<double> costs;

  std::cout << "Initial thetas" << std::endl;
  print(t);

  costs = logisticRegressionCosts(X, t);
  costBefore = logisticCostFn(X, Y, costs);

  bool continueFlag = false;

  auto start = std::chrono::high_resolution_clock::now();
  do
  {
    logisticThetasUpdater(X, Y, t, alpha, costs);

    costs = logisticRegressionCosts(X, t);
    costAfter = logisticCostFn(X, Y, costs);
    if (iterationCount % 10000 == 0)
    {
      std::cout << "thetas: ";
      print(t);
    }
    iterationCount++;

    // std::cout << "Cost before: " << costBefore << " cost after: " << costAfter << std::endl;

    continueFlag = costAfter < costBefore;
    costBefore = costAfter;
    if (isEqual(costAfter, 0))
      continueFlag = false;
  } while (continueFlag);

  std::cout << "Cost before: " << costBefore << " cost after: " << costAfter << std::endl;
  std::cout << "Iteration count: " << iterationCount << std::endl;
  std::cout << "Last alpha: " << alpha << std::endl;
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Time elapsed ms: " << elapsed.count() << std::endl;
  return t;
}

void logisticRegressionCodeIterative(const featuresSet &x, const resultsSet &y, const double alpha)
{
  std::cout << "Logistic version of gradient descent" << std::endl;
  int iterationCount = 0;
  std::vector<double> costs;
  std::vector<double> thetas(x[0].size(), 0);

  std::cout << "X matrix" << std::endl;
  for (int i = 0; i < x.size(); ++i)
  {
    print(x[i]);
  }

  std::cout << std::endl;
  std::cout << "Y vector" << std::endl;
  print(y);

  std::vector<double> calculatedThetas;
  // thetas = {-20.45017846, 0.16858011, 0.16333761};
  calculatedThetas = gradientDescentLogisticVersion(x, y, thetas, alpha);
  std::cout << "Calculated thetas" << std::endl;
  print(calculatedThetas);
  std::cout << std::endl
            << std::endl;

  std::vector<double> testX1 = {1, 74.7759, 89.5298}; // should be 1
  std::vector<double> testX2 = {1, 34.6237, 78.0247}; // should be 0
  std::cout << "Test values: " << std::endl;
  std::cout << "testX1: " << logisticH(testX1, calculatedThetas) << std::endl;
  std::cout << "testX2: " << logisticH(testX2, calculatedThetas) << std::endl;
}

#endif