#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <cassert>
#include <cmath>
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

// double G(double HResult)
// {
//   // std::cout << "OT: " << HResult << " e^-(OT): " << std::exp(-HResult) << std::endl;

//   return 1.0 / (1.0 + std::exp(-HResult));
// }

double logisticCostFn(const featuresSet &x, const resultsSet &y, const std::vector<double> &thetas)
{
  double cost = 0.0;
  double g = 0.0;
  double l = 0.0;
  double lg = 0.0;
  int m = x.size();
  for (int i = 0; i < m; ++i)
  {
    // std::cout << "LogisticH: "
    //           << " [ " << y[i] << " ] " << logisticH(x[i], thetas) << std::endl;
    g = logisticH(x[i], thetas);
    // std::cout << "g: " << g << std::endl;
    cost += y[i] * std::log(g) + (1 - y[i]) * std::log(1 - g);
  }

  // std::cout << "Cost: " << cost << std::endl;

  return -1 * cost / m;
}

std::vector<double> logisticRegressionCosts(const featuresSet &x, const std::vector<double> &y, const std::vector<double> &thetas)
{
  int m = x.size();
  std::vector<double> costs;
  for (int i = 0; i < m; ++i)
  {
    costs.push_back(logisticH(x[i], thetas) - y[i]);
  }
  return costs;
}

void logisticThetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha, const std::vector<double> &costs)
{
  std::vector<double> tmpThetas;

  int m = x.size();
  int n = thetas.size();
  double prediction = 0.0;
  std::vector<double> grad;
  for (int j = 0; j < n; ++j)
  {
    prediction = 0.0;
    // INSIDE SUM i=0 to m
    for (int i = 0; i < m; ++i)
    {
      prediction += costs[i] * x[i][j];
    }
    // grad.push_back((1.0 / m) * prediction);
    tmpThetas.push_back(thetas[j] - ((alpha / m) * prediction)); // (alpha / m)
  }
  // std::cout << "Grad: " << std::endl;
  // print(grad);
  thetas = tmpThetas;
}

std::vector<double> gradientDescentLogisticVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, const double alpha)
{
  double firstCost = 0.0;
  double initialCost = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;

  std::vector<double> t = thetas;

  std::cout << "Initial thetas" << std::endl;
  print(t);
  firstCost = logisticCostFn(X, Y, t);
  std::cout << "First cost: " << firstCost << std::endl;
  std::vector<double> costs;
  do
  {
    initialCost = logisticCostFn(X, Y, t);
    costs = logisticRegressionCosts(X, Y, t);
    logisticThetasUpdater(X, Y, t, alpha, costs);
    // std::cout << "First updated thetas" << std::endl;
    // print(t);
    costAfter = logisticCostFn(X, Y, t);
    if (iterationCount % 10000 == 0)
    {
      std::cout << "It: " << iterationCount << std::endl;
      print(t);
    }
    iterationCount++;
    if (isEqual(costAfter, initialCost) || isEqual(costAfter, 0))
      break;
  } while (costAfter < initialCost);
  std::cout << "Initial cost: " << initialCost << " cost after: " << costAfter << std::endl;
  std::cout << "Iteration count: " << iterationCount << std::endl;
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
  thetas = {-20.45017846, 0.16858011, 0.16333761};
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