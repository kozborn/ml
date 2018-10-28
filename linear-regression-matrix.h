#ifndef LINEAR_REGRESSION_MATRIX
#define LINEAR_REGRESSION_MATRIX

#include <iostream>
#include <fstream>
#include <vector>
#include "utils.h"
#include "martix_ml_utils.h"
#include <Eigen/Dense>

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
