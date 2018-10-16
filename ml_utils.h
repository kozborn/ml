#ifndef ML_UTILS_H
#define ML_UTILS_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "utils.h"

typedef std::vector<double> featuresRow;
typedef std::vector<featuresRow> featuresSet;

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
  for (int t = 0; t < thetas.size(); ++t)
  {
    tmpSum = 0.0;
    for (int i = 0; i < x[t].size(); ++i)
    {
      tmpSum += (H(thetas, x[i]) - y[i]) * x[t][i];
    }
    tmpThetas.push_back(thetas[t] - ((alpha / (2 * x[t].size())) * tmpSum));
  }

  thetas = tmpThetas;
}

void scaleFeatures(featuresSet &x, double min, double max, bool printScalingResults = false)
{
  if (printScalingResults)
  {
    std::cout << std::endl
              << "before scaling" << std::endl
              << std::endl;
    for (int i = 0; i < x.size(); ++i)
    {
      print(x[i]);
    }
  }

  double avg = (max - min) / 2;

  for (int i = 0; i < x.size(); ++i)
  {
    for (int j = 0; j < x[i].size(); ++j)
    {
      x[i][j] = (x[i][j] - avg) / (max - min);
    }
  }

  if (printScalingResults)
  {
    std::cout << std::endl
              << "after scaling" << std::endl;
    for (int i = 0; i < x.size(); ++i)
    {
      print(x[i]);
    }
    std::cout << std::endl
              << "end of scaling" << std::endl;
  }
}

#endif
