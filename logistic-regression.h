#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <cassert>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <queue>
#include "utils.h"
#include "ml_utils.h"
#include "threadpool.h"

std::mutex derivative_mutex;
std::mutex theta_mutex;
std::mutex output_mutex;

int threadCounter = 0;

void printClassAndThetas(int, std::vector<double>);

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

void derivative(const featuresSet &x, const resultsSet &y, const std::vector<double> &costs, const double alpha, const double theta, const double currentClass, std::vector<double> &result, const int paramNum)
{
  int m = x.size();
  double tmpY = 0.0;
  double der = 0.0;
  // INSIDE SUM i=0 to m
  for (int i = 0; i < m; ++i)
  {
    tmpY = y[i] == currentClass ? 1 : 0;
    der += (costs[i] - tmpY) * x[i][paramNum];
  }
  result[paramNum] = theta - ((alpha / m) * der);
  // std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void logisticThetasUpdater(const featuresSet &x, const std::vector<double> &y, std::vector<double> &thetas, const double alpha, const std::vector<double> &costs, double currentClass = 1.0)
{
  try
  {
    std::vector<double> tmpThetas(thetas.size());
    int n = thetas.size();
    std::vector<std::thread> derThreads;
    for (int j = 0; j < n; ++j)
    {
      derivative(std::ref(x), std::ref(y), std::ref(costs), alpha, thetas[j], currentClass, std::ref(tmpThetas), j);
      threadCounter++;
    }

    for (int j = 0; j < derThreads.size(); ++j)
    {
      if (derThreads[j].joinable())
        derThreads[j].join();
    }
    thetas = tmpThetas;
  }
  catch (...)
  {
    std::cout << "Sth is fucked up" << std::endl;
  }
}

std::vector<double> multiclassGradientDescentLogisticVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, double alpha, double currentClass)
{
  double costBefore = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;
  const double errorMargin = 0.00000000001;

  std::vector<double> t = thetas;
  std::vector<double> costs;

  // std::cout << "Initial thetas ";
  // print(t);

  costs = logisticRegressionCosts(X, t);
  costBefore = logisticCostFn(X, Y, costs, currentClass);

  bool continueFlag = false;
  auto start = std::chrono::high_resolution_clock::now();
  do
  {
    theta_mutex.lock();
    logisticThetasUpdater(X, Y, t, alpha, costs, currentClass);
    costs = logisticRegressionCosts(X, t);
    costAfter = logisticCostFn(X, Y, costs, currentClass);
    if (iterationCount % 5000 == 0 && iterationCount != 0)
    {
      std::cout << "Class (" << currentClass << ") Thetas: ";
      print(t);
    }
    continueFlag = costAfter < costBefore;

    if (isEqual(costAfter, 0))
      continueFlag = false;

    if (std::abs(costBefore - costAfter) < errorMargin)
      continueFlag = false;
    // std::cout << iterationCount << std::endl;
    costBefore = costAfter;
    theta_mutex.unlock();
    iterationCount++;
  } while (iterationCount < 1000000);

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Time elapsed ms: " << elapsed.count() << std::endl;
  return t;
}

void multiThreadedGradientDescent(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, double alpha, double currentClass, std::vector<double> &output)
{
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
    // std::cout << "Current class: " << possibleOutputs[c] << std::endl;
    // std::cout << x.size() << " " << y.size() << " " << readyThetas[c].size() << " " << alpha << " " << possibleOutputs[c] << " " << calculatedThetas[c].size() << std::endl;
    // multiThreadedGradientDescent(std::ref(x), std::ref(y), readyThetas[c], alpha, possibleOutputs[c], std::ref(calculatedThetas[c]));
    std::thread grad(multiThreadedGradientDescent, std::ref(x), std::ref(y), readyThetas[c], alpha, possibleOutputs[c], std::ref(calculatedThetas[c]));
    threads.push_back(move(grad));
  }

  for (int t = 0; t < threads.size(); ++t)
  {
    threads[t].join();
  }

  for (int c = 0; c < possibleOutputs.size(); ++c)
  {
    printClassAndThetas(possibleOutputs[c], calculatedThetas[c]);
  }

  // Saving calculated thetas to ftile
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

void printClassAndThetas(int cl, std::vector<double> t)
{
  std::cout << "Class: " << cl << " Calculated thetas ";
  print(t);
  std::cout << std::endl;
}

#endif