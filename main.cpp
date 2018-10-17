#include <algorithm>
#include <cmath>
#include <ctime>
#include <exception>
#include <fstream>
#include <limits>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include "ml_utils.h"
#include "utils.h"

std::vector<double> y;
std::string octaveHeader = "# Created by Octave 3.8.2, Fri Oct 12 16:32:14 2018 CEST <piotrkozubek@gmail.com>";

featuresSet x;

// Theta updater from vectorization = theta = theta - alpha/m * (X' * (X*theta - Y))

int main()
{
  try
  {
    std::srand(time(nullptr));
    std::vector<double> thetas = {0, 1, 3};
    int featuresSize = thetas.size() - 1;
    double alpha = 0.005;
    double errorMargin = 0.05;
    int trainingSampleSize = 10;
    std::vector<double> HVector;

    std::cout << "Squared cost function v.0.0.1" << std::endl;
    featuresRow row;

    for (int k = 0; k < trainingSampleSize; ++k)
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

    for (int i = 0; i < trainingSampleSize; ++i)
    {
      y.push_back(i + 2);
      // print(x[i]);
    }
    std::cout << std::endl;
    print(y);

    // fillVector(y, trainingSampleSize, 0, 20);
    // std::sort(y.begin(), y.end());

    int iterationCount = 0;
    double costAfter = 0.0;
    double initialCost = 0.0;

    // std::cout << "Initial cost: " << costFn(thetas, x, y) << std::endl;
    do
    {
      initialCost = costFn(thetas, x, y);
      thetasUpdater(thetas, alpha, x, y);
      costAfter = costFn(thetas, x, y);
      iterationCount++;
      if (isEqual(costAfter, initialCost) || isEqual(costAfter, 0))
        break;
    } while (costAfter < initialCost);
    std::cout << "Thetas after convergence" << std::endl;
    print(thetas);
    std::cout << "Cost " << costFn(thetas, x, y) << std::endl;
    std::cout << "y vector" << std::endl;
    print(y);

    for (int k = 0; k < x.size(); ++k)
    {
      HVector.push_back(H(thetas, x[k]));
    }

    std::cout << "h vector" << std::endl;
    print(HVector);
    std::cout << "Iteration count " << iterationCount << std::endl;

    // SAVING FILE FOR OCTAVE

    // std::ofstream outputFile;
    // outputFile.open("data.mat");
    // if (outputFile.good())
    // {
    //   outputFile << octaveHeader << std::endl;
    //   outputFile << vectorToOctaveFormat("x", x) << std::endl;
    //   outputFile << vectorToOctaveFormat("y", y) << std::endl;
    //   outputFile << vectorToOctaveFormat("h", HVector) << std::endl;
    // }
  }
  catch (std::exception &e)
  {
    std::cerr << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "Ups. Something went wrong" << std::endl;
  }

  getchar();
  return 0;
}
