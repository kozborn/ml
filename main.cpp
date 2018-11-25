#include <algorithm>
#include <cmath>
#include <ctime>
#include <exception>
#include <fstream>
#include <limits>
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "ml_utils.h"
#include "utils.h"
#include "linear-regression.h"
#include "logistic-regression.h"

// std::string featuresFile = "./sample_data/sample-features-logistic.txt";
// std::string resultsFile = "./sample_data/sample-results-logistic.txt";
std::string featuresFile = "./sample_data/coursera-features-set1.txt";
std::string resultsFile = "./sample_data/multiclass-coursera-results-set.txt";

// std::string featuresFile = "./machine-learning-ex2/ex2/x.txt";
// std::string resultsFile = "./machine-learning-ex2/ex2/y.txt";

double alpha = 0.0005;

featuresSet x;
resultsSet y;
int main(int argc, char *argv[])
{
  try
  {
    std::cout << "Machine learning - v.0.0.4" << std::endl;
    if (argc > 3)
    {
      featuresFile = argv[1];
      resultsFile = argv[2];
      sscanf(argv[3], "%lf", &alpha);
    }
    std::cout << "File with features data: " << featuresFile << std::endl;
    std::cout << "File with results data: " << resultsFile << std::endl;
    std::cout << "Starting with alpha: " << alpha << std::endl;
    readMatrixFromFile(featuresFile, x);
    readVectorFromFile(resultsFile, y);
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
