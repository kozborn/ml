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
    std::cout << "Machine learning - v.0.0.3 - multiclass classification" << std::endl;
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

    // GENERATE RESULTS SET in range 1 - 3
    // int num = 0;
    // std::ofstream outputFile;
    // outputFile.open("./sample_data/multiclass-coursera-results-set.txt");
    // if (outputFile.good())
    // {

    //   for (int i = 0; i < x.size(); ++i)
    //   {
    //     if (x[i][0] > 75 && x[i][1] < 75)
    //       num = 1;
    //     else if (x[i][0] < 75 && x[i][1] > 75)
    //       num = 2;
    //     else
    //       num = 3;
    //     // num = random(1, 3);
    //     std::cout << num << std::endl;
    //     outputFile << num << std::endl;
    //   }
    // }
    // outputFile.close();

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
