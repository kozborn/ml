#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include "utils.h"
#include "ml_utils.h"

std::vector<double> gradientDescentIterativeVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, const double alpha)
{
  double firstCost = 0.0;
  double initialCost = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;
  std::vector<double> costs;
  std::vector<double> t = thetas;
  firstCost = costFn(X, Y, t);
  do
  {
    initialCost = costFn(X, Y, t);
    costs = linearRegressionCosts(X, Y, t);
    thetasUpdater(X, Y, t, alpha, costs);
    costAfter = costFn(X, Y, t);

    iterationCount++;
    if (isEqual(costAfter, initialCost) || isEqual(costAfter, 0))
      break;
  } while (costAfter < initialCost);
  std::cout << "Initial cost: " << firstCost << " cost after: " << costAfter << std::endl;
  std::cout << "Iteration count: " << iterationCount << std::endl;
  return t;
}

void linearRegressionCodeIterative(const featuresSet &x, const resultsSet &y, double alpha)
{
  std::cout << "Iterative version of gradient descent" << std::endl;
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
  calculatedThetas = gradientDescentIterativeVersion(x, y, thetas, alpha);
  std::cout << "Calculated thetas" << std::endl;
  print(calculatedThetas);
  std::cout << std::endl
            << std::endl;
}

void featureScale(featuresSet &x)
{
  double min = 0.0;
  double max = 0.0;
  std::vector<double> mins;
  std::vector<double> maxes;
  std::vector<double> tmp;

  for (int j = 0; j < x[0].size(); ++j)
  {
    tmp.clear();
    for (int i = 0; i < x.size(); ++i)
    {
      tmp.push_back(x[i][j]);
    }
    mins.push_back(*std::min_element(tmp.begin(), tmp.end()));
    maxes.push_back(*std::max_element(tmp.begin(), tmp.end()));
  }

  std::cout << "Mins" << std::endl;
  print(mins);
  std::cout << "Maxes" << std::endl;
  print(maxes);

  scaleFeatures(x, mins, maxes);
}

#endif
