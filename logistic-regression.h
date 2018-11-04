#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <cassert>
#include <cmath>
#include <chrono>
#include <fstream>
#include <set>
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

double logisticCostFn(const featuresSet &x, const resultsSet &y, const std::vector<double> &costs, double currentClass = 1.0)
{
  double cost = 0.0;
  double g = 0.0;
  int m = x.size();
  double tmpY = 0.0;
  for (int i = 0; i < m; ++i)
  {
    tmpY = y[i] == currentClass ? 1 : 0;
    g = costs[i];
    cost += tmpY * std::log(g) + (1 - tmpY) * std::log(1 - g);
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

void logisticThetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha, const std::vector<double> &costs, double currentClass = 1.0)
{
  std::vector<double> tmpThetas;

  int m = x.size();
  int n = thetas.size();
  double prediction = 0.0;
  double tmpY = 0.0;
  for (int j = 0; j < n; ++j)
  {
    prediction = 0.0;
    // INSIDE SUM i=0 to m
    for (int i = 0; i < m; ++i)
    {
      tmpY = y[i] == currentClass ? 1 : 0;
      prediction += (costs[i] - tmpY) * x[i][j];
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
  const double errorMargin = 0.1;

  std::vector<double> t = thetas;
  std::vector<double> costs;

  std::cout << "Initial thetas" << std::endl;
  print(t);

  costs = logisticRegressionCosts(X, t);
  costBefore = logisticCostFn(X, Y, costs);

  bool continueFlag = false;
  std::vector<double> testValues1 = {1, 75.0247, 46.5540}; // should be 1
  std::vector<double> testValues2 = {1, 34.6237, 78.0247}; // should be 2
  std::vector<double> testValues3 = {1, 30.2867, 43.8950}; // should be 3

  auto start = std::chrono::high_resolution_clock::now();

  do
  {
    logisticThetasUpdater(X, Y, t, alpha, costs);

    costs = logisticRegressionCosts(X, t);
    costAfter = logisticCostFn(X, Y, costs);
    if (iterationCount % 50000 == 0)
    {
      std::cout << "thetas: ";
      print(t);
      std::cout << "P(y = 1): " << std::abs(logisticH(testValues1, t)) << std::endl;
      std::cout << "P(y = 2): " << std::abs(logisticH(testValues2, t)) << std::endl;
      std::cout << "P(y = 3): " << std::abs(logisticH(testValues3, t)) << std::endl;
      std::cout << std::endl;
    }
    iterationCount++;

    continueFlag = costAfter < costBefore;
    costBefore = costAfter;
    if (isEqual(costAfter, 0))
      continueFlag = false;

    // if (std::abs(logisticH(testValues1, t) - 1) < errorMargin && std::abs(logisticH(testValues2, t) < errorMargin))
    //   continueFlag = false;

  } while (continueFlag);

  std::cout << "Cost before: " << costBefore << " cost after: " << costAfter << std::endl;
  std::cout << "Iteration count: " << iterationCount << std::endl;
  std::cout << "Last alpha: " << alpha << std::endl;
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Time elapsed ms: " << elapsed.count() << std::endl;
  return t;
}

std::vector<double> multiclassGradientDescentLogisticVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, double alpha, double currentClass)
{
  double costBefore = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;
  const double errorMargin = 0.00000000001;

  std::vector<double> t = thetas;
  std::vector<double> costs;

  std::cout << "Initial thetas" << std::endl;
  print(t);

  costs = logisticRegressionCosts(X, t);
  costBefore = logisticCostFn(X, Y, costs, currentClass);

  bool continueFlag = false;
  auto start = std::chrono::high_resolution_clock::now();
  do
  {
    logisticThetasUpdater(X, Y, t, alpha, costs, currentClass);

    costs = logisticRegressionCosts(X, t);
    costAfter = logisticCostFn(X, Y, costs, currentClass);
    if (iterationCount % 50000 == 0 && iterationCount != 0)
    {
      std::cout << "Class (" << currentClass << ") "
                << "It: " << iterationCount << " Thetas: ";
      print(t);
      std::cout << std::endl;
    }
    continueFlag = costAfter < costBefore;

    if (isEqual(costAfter, 0))
      continueFlag = false;

    if (std::abs(costBefore - costAfter) < errorMargin)
      continueFlag = false;

    costBefore = costAfter;
    iterationCount++;
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

  std::set<double> classess(y.begin(), y.end());
  std::vector<double> possibleOutputs(classess.begin(), classess.end());
  std::cout << "Classess" << std::endl;
  print(possibleOutputs);
  std::cout << std::endl;

  int iterationCount = 0;
  std::vector<double> thetas(x[0].size(), 0);

  // std::cout << "X matrix" << std::endl;
  // for (int i = 0; i < x.size(); ++i)
  // {
  //   print(x[i]);
  // }

  // std::cout << std::endl;
  // std::cout << "Y vector" << std::endl;
  // print(y);

  std::vector<std::vector<double>> readyThetas = {
      {-10.763, 0.234, -0.132},
      {-11.176, -0.123, 0.231},
      {3.953, -0.034, -0.253}};

  double currentClass = 0.0;
  std::vector<std::vector<double>> calculatedThetas(possibleOutputs.size());
  for (int c = 0; c < possibleOutputs.size(); ++c)
  {
    std::cout << "Current class: " << possibleOutputs[c] << std::endl;
    calculatedThetas[c] = multiclassGradientDescentLogisticVersion(x, y, readyThetas[c], alpha, possibleOutputs[c]);
    std::cout << std::endl;
  }

  for (int c = 0; c < possibleOutputs.size(); ++c)
  {
    std::cout << "Calculated thetas" << std::endl;
    print(calculatedThetas[c]);
    std::cout << std::endl;
  }
  // Saving calculated thetas to file
  std::ofstream outputThetasFile;
  outputThetasFile.open("thetas.txt");
  if (outputThetasFile.good())
  {
    for (int i = 0; i < calculatedThetas.size(); ++i)
    {
      for (int k = 0; k < calculatedThetas[i].size(); ++k)
      {
        outputThetasFile << calculatedThetas[i][k] << " ";
      }
      outputThetasFile << std::endl;
    }
  }

  std::vector<double> t1 = {1, 75.0247, 46.5540}; // should be 1
  std::vector<double> t2 = {1, 34.6237, 78.0247}; // should be 2
  std::vector<double> t3 = {1, 30.2867, 43.8950}; // should be 3
  std::vector<std::vector<double>> testParams = {t1, t2, t3};

  for (int p = 0; p < calculatedThetas.size(); ++p)
  {
    std::cout << "P(y=1) : " << logisticH(testParams[0], calculatedThetas[p]) << std::endl;
    std::cout << "P(y=2) : " << logisticH(testParams[1], calculatedThetas[p]) << std::endl;
    std::cout << "P(y=3) : " << logisticH(testParams[2], calculatedThetas[p]) << std::endl;
    std::cout << std::endl;
  }
}

#endif