#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include "utils.h"
#include <sstream>
#include "ml_utils.h"

void loadMatrix(std::istream &is, std::vector<std::vector<double>> &matrix, const std::string &delimeter)
{
  std::string line;
  std::string strnum;

  matrix.clear();

  while (std::getline(is, line))
  {
    matrix.push_back(std::vector<double>());
    for (std::string::const_iterator i = line.begin(); i != line.end(); i++)
    {
      if (delimeter.find(*i) == std::string::npos)
      {
        strnum += *i;
        if (i + 1 != line.end())
          continue;
      }
      if (strnum.empty())
        continue;

      double number;
      std::istringstream(strnum) >> number;
      matrix.back().push_back(number);
      strnum.clear();
    }
  }
}

void readMatrixFromFile(const std::string filename, featuresSet &X)
{
  std::cout << "Reading matrix from file: " << filename << std::endl;
  std::ifstream inputFile;

  inputFile.open(filename);
  if (inputFile.good())
  {
    loadMatrix(inputFile, X, " ");
  }
  else
  {
    std::cerr << "Cannot open file" << filename << std::endl;
  }
}

void readVectorFromFile(const std::string filename, resultsSet &Y)
{
  std::cout << "Reading vector from file: " << filename << std::endl;
  std::ifstream inputFile;
  double y = 0.0;
  inputFile.open(filename);
  if (inputFile.good())
  {
    while (inputFile >> y)
    {
      Y.push_back(y);
    }
  }
  else
  {
    std::cerr << "Cannot open file" << filename << std::endl;
  }
}

std::vector<double> gradientDescentIterativeVersion(const featuresSet &X, const resultsSet &Y, const std::vector<double> thetas, const double alpha)
{
  double initialCost = 0.0;
  double costAfter = 0.0;
  int iterationCount = 0;

  std::vector<double> t = thetas;
  do
  {
    initialCost = costFn(X, Y, t);

    thetasUpdater(X, Y, t, alpha);
    costAfter = costFn(X, Y, t);

    iterationCount++;
    if (isEqual(costAfter, initialCost) || isEqual(costAfter, 0))
      break;
  } while (costAfter < initialCost);
  std::cout << "Initial cost: " << initialCost << " cost after: " << costAfter << std::endl;
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

  std::vector<double> calculatedThetas;
  calculatedThetas = gradientDescentIterativeVersion(x, y, thetas, alpha);
  std::cout << "Calculated thetas" << std::endl;
  print(calculatedThetas);
  std::cout << std::endl
            << std::endl;
}

void append1toFeaturesSet(featuresSet &x)
{
  // Inserting 1 at the first column of training set
  for (int i = 0; i < x.size(); ++i)
  {
    x[i].insert(x[i].begin(), 1);
  }
}

void featureScale(featuresSet &x)
{
  double min = 0.0;
  double max = 0.0;
  std::vector<double> mins;
  std::vector<double> maxes;
  std::vector<double> tmp;

  //TODO odwrócić kolejność iteracji, najpierw iterować po j jako liczbie column
  // później po i jako wiersze

  for (std::vector<std::vector<double>>::iterator it = x.begin(); it != x.end(); ++it)
  {
    tmp = *it;
    std::cout << tmp[0] << std::endl;
  }
  {

    // TODO this has to be done for each feature column
    // double xMax = *std::max_element(tempXVector.begin(), tempXVector.end());
    // double xMin = *std::min_element(tempXVector.begin(), tempXVector.end());
  }
  // scaleFeatures(x, mins, maxes);
}

#endif
