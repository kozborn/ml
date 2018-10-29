#ifndef MATRIX_ML_UTILS_H
#define MATRIX_ML_UTILS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "utils.h"
#include <Eigen/Dense>

typedef std::vector<double> resultsSet;
typedef std::vector<double> featuresRow;
typedef std::vector<featuresRow> featuresSet;

// Theta updater from vectorization = theta = theta - alpha/m * (X' * (X*theta - Y))
Eigen::MatrixXd normalEquation(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
  return (X.transpose() * X).completeOrthogonalDecomposition().pseudoInverse() * X.transpose() * Y;
}

#endif