#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

#include <iostream>
#include <vector>
#include "utils.h"
#include "ml_utils.h"

std::vector<double> gradientDescentIterativeVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, const double alpha)
{
  double initialCost = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;
  std::vector<double> costs;
  std::vector<double> t = thetas;
  do
  {
    initialCost = costFn(t, X, Y);
    costs = linearRegressionCosts(X, Y, t);
    thetasUpdater(X, Y, t, alpha, costs);
    costAfter = costFn(t, X, Y);
    iterationCount++;
    if (isEqual(costAfter, initialCost) || isEqual(costAfter, 0))
      break;
  } while (costAfter < initialCost);
  std::cout << "Iteration count: " << iterationCount << std::endl;
  return t;
}

void linearRegressionCodeIterative()
{
  std::vector<double> y;
  std::string octaveHeader = "# Created by Octave 3.8.2, Fri Oct 12 16:32:14 2018 CEST <piotrkozubek@gmail.com>";
  featuresSet x;
  std::cout << "Iterative version of gradient descent" << std::endl;
  std::vector<double> thetas = {0, 0, 0, 0, 0};
  int m = 5; // training sets
  int n = 5; // features size
  double alpha = 0.0005;
  int featuresSize = thetas.size() - 1;
  std::vector<double> HVector;

  featuresRow row;

  for (int k = 0; k < m; ++k)
  {
    row.clear();
    row.push_back(1);
    for (int f = 0; f < featuresSize; ++f)
    {
      row.push_back(k + 1);
    }
    x.push_back(row);
  }

  // double min = 1;
  // double max = 10;
  // scaleFeatures(x, min, max);

  for (int i = 0; i < m; ++i)
  {
    y.push_back(i + 2);
    print(x[i]);
  }
  std::cout << std::endl;
  print(y);

  std::vector<double> calculatedThetas;
  calculatedThetas = gradientDescentIterativeVersion(x, y, thetas, alpha);
  std::cout << "Calculated thetas" << std::endl;
  print(calculatedThetas);
  std::cout << std::endl
            << std::endl;
}

void linearRegressionCodeVectorized()
{
  std::cout << "Vectorized Gradient descent" << std::endl;
  int m = 5; // training sets
  int n = 5; // features size

  Eigen::MatrixXd theta(m, 1);
  Eigen::MatrixXd X(m, n);
  Eigen::MatrixXd Y(m, 1);

  for (int i = 0; i < m; ++i)
  {
    Y(i, 0) = i + 2;
    for (int j = 0; j < n; ++j)
    {
      X(i, j) = i + 1;
    }
    X(i, 0) = 1;
  }

  std::cout << std::endl;
  std::cout << Y << std::endl;
  std::cout << std::endl;
  std::cout << X << std::endl;
  std::cout << std::endl;
  theta = normalEquation(X, Y);
  std::cout << "Calculated thetas" << std::endl;
  std::cout << theta << std::endl;
  std::cout << std::endl
            << std::endl;
}

#endif
