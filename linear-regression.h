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
  std::vector<double> t = thetas;
  do
  {
    initialCost = costFn(t, X, Y);
    thetasUpdater(t, alpha, X, Y);
    costAfter = costFn(t, X, Y);
    iterationCount++;
    if (isEqual(costAfter, initialCost) || isEqual(costAfter, 0))
      break;
  } while (costAfter < initialCost);

  return t;
}

#endif
