#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <cassert>
#include <cmath>
#include <chrono>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include "utils.h"
#include "ml_utils.h"

std::mutex derivative_mutex;

double logisticH(const std::vector<double> &features, const std::vector<double> &thetas)
{
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

void derivative(const featuresSet &x, const resultsSet &y, const std::vector<double> &costs, const double alpha, const double theta, const double currentClass, double &result, const int paramNum)
{
  int m = x.size();
  double tmpY = 0.0;
  double der = 0.0;
  // INSIDE SUM i=0 to m
  // derivative_mutex.lock();
  for (int i = 0; i < m; ++i)
  {
    tmpY = y[i] == currentClass ? 1 : 0;
    der += (costs[i] - tmpY) * x[i][paramNum];
  }
  result = theta - ((alpha / m) * der);
  // derivative_mutex.unlock();
}

void logisticThetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha, const std::vector<double> &costs, double currentClass = 1.0)
{
  std::vector<double> tmpThetas(thetas.size());

  int n = thetas.size();

  std::vector<std::thread> derThreads;
  for (int j = 0; j < n; ++j)
  {
    // std::cout << "Creating thread " << j << std::endl;
    std::thread th(derivative, std::ref(x), std::ref(y), std::ref(costs), alpha, thetas[j], currentClass, std::ref(tmpThetas[j]), j);
    // derivative(std::ref(x), std::ref(y), std::ref(costs), alpha, thetas[j], currentClass, std::ref(tmpThetas[j]), j);
    derThreads.push_back(move(th));
  }
  for (int j = 0; j < derThreads.size(); ++j)
  {
    std::cout << "Joining thread " << j << std::endl;
    derThreads[j].join();
  }
  thetas = tmpThetas;
}

// std::vector<double> gradientDescentLogisticVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, double alpha)
// {
//   double costBefore = 0.0;
//   double costAfter = 0.0;
//   int iterationCount = 0;
//   const double errorMargin = 0.1;

//   std::vector<double> t = thetas;
//   std::vector<double> costs;

//   std::cout << "Initial thetas" << std::endl;
//   print(t);

//   costs = logisticRegressionCosts(X, t);
//   costBefore = logisticCostFn(X, Y, costs);

//   bool continueFlag = false;

//   auto start = std::chrono::high_resolution_clock::now();

//   do
//   {
//     logisticThetasUpdater(X, Y, t, alpha, costs);
//     std::cout << "Thetas: ";
//     print(t);
//     costs = logisticRegressionCosts(X, t);
//     costAfter = logisticCostFn(X, Y, costs);

//     iterationCount++;
//     continueFlag = costAfter < costBefore;
//     costBefore = costAfter;
//     if (isEqual(costAfter, 0))
//       continueFlag = false;

//   } while (continueFlag);

//   std::cout << "Cost before: " << costBefore << " cost after: " << costAfter << std::endl;
//   std::cout << "Iteration count: " << iterationCount << std::endl;
//   std::cout << "Last alpha: " << alpha << std::endl;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Time elapsed ms: " << elapsed.count() << std::endl;
//   return t;
// }

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
      std::cout << "Class (" << currentClass << ") Thetas: ";
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

void multiThreadedGradientDescent(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, double alpha, double currentClass, std::vector<double> &output)
{
  std::cout << X.size() << " " << Y.size() << " " << thetas.size() << " " << alpha << " " << currentClass << " " << output.size() << std::endl;
  output = multiclassGradientDescentLogisticVersion(X, Y, thetas, alpha, currentClass);
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

  std::vector<std::vector<double>> readyThetas = {
      {-10.763, 0.234, -0.132},
      {-11.176, -0.123, 0.231},
      {3.953, -0.034, -0.253}};

  double currentClass = 0.0;

  std::vector<std::vector<double>> calculatedThetas(possibleOutputs.size());
  std::vector<std::thread> threads;

  for (int c = 0; c < possibleOutputs.size(); ++c)
  {
    std::cout << "Current class: " << possibleOutputs[c] << std::endl;
    std::cout << x.size() << " " << y.size() << " " << readyThetas[c].size() << " " << alpha << " " << possibleOutputs[c] << " " << calculatedThetas[c].size() << std::endl;
    // std::thread grad(multiThreadedGradientDescent, std::ref(x), std::ref(y), readyThetas[c], alpha, possibleOutputs[c], std::ref(calculatedThetas[c]));
    multiThreadedGradientDescent(x, y, readyThetas[c], alpha, possibleOutputs[c], calculatedThetas[c]);
    // threads.push_back(move(grad));
  }

  for (int t = 0; t < threads.size(); ++t)
  {
    threads[t].join();
  }

  for (int c = 0; c < possibleOutputs.size(); ++c)
  {
    std::cout << "Calculated thetas" << std::endl;
    print(calculatedThetas[c]);
    std::cout << std::endl;
  }

  // Saving calculated thetas to file
  // std::ofstream outputThetasFile;
  // outputThetasFile.open("thetas.txt");
  // if (outputThetasFile.good())
  // {
  //   for (int i = 0; i < calculatedThetas.size(); ++i)
  //   {
  //     for (int k = 0; k < calculatedThetas[i].size(); ++k)
  //     {
  //       outputThetasFile << calculatedThetas[i][k] << " ";
  //     }
  //     outputThetasFile << std::endl;
  //   }
  // }
}

#endif