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
std::string resultsFile = "./sample_data/coursera-results-set1.txt";

// std::string featuresFile = "./machine-learning-ex2/ex2/x.txt";
// std::string resultsFile = "./machine-learning-ex2/ex2/y.txt";

double alpha = 0.5;

featuresSet x;
resultsSet y;
int main(int argc, char *argv[])
{
  try
  {
    std::cout << "Machine learning - v.0.0.2 - classification" << std::endl;
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
    // std::cout << "Running linear regressiong code iterative version" << std::endl;
    // // featureScale(x);
    // append1toFeaturesSet(x);

    // linearRegressionCodeIterative(x, y, alpha);
    // TODO create also vectorized implementation
    // linearRegressionCodeVectorized();

    // LOGISTIC REGRESSION
    std::cout << "Running logistic regressiong code iterative version" << std::endl;
    append1toFeaturesSet(x);
    logisticRegressionCodeIterative(x, y, alpha);
    // std::cout << "Costs" << std::endl;
    // print(costs);
    // std::cout << std::endl;
    // std::cout << alpha << std::endl;

    // thetasUpdater(x, y, thetas, alpha, costs);
    // std::cout << "Thetas" << std::endl;
    // print(thetas);
    // std::cout << std::endl;

    // // SAVING FILE FOR OCTAVE

    // // std::ofstream outputFile;
    // // outputFile.open("data.mat");
    // // if (outputFile.good())
    // // {
    // //   outputFile << octaveHeader << std::endl;
    // //   outputFile << vectorToOctaveFormat("x", x) << std::endl;
    // //   outputFile << vectorToOctaveFormat("y", y) << std::endl;
    // //   outputFile << vectorToOctaveFormat("h", HVector) << std::endl;
    // // }
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
