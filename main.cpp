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
#include <Eigen/Dense>
#include "ml_utils.h"
#include "utils.h"
#include "linear-regression.h"

std::vector<double> y;
std::string octaveHeader = "# Created by Octave 3.8.2, Fri Oct 12 16:32:14 2018 CEST <piotrkozubek@gmail.com>";

featuresSet x;

int main()
{
  try
  {
    std::cout << "Squared cost function v.0.0.1" << std::endl;
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

    std::cout << "Iterative version of gradient descent" << std::endl;
    std::srand(time(nullptr));
    std::vector<double> thetas = {0, 0, 0, 0, 0};
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
