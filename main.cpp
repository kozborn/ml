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

std::string featuresFile = "./sample_data/sample-features.txt";
std::string resultsFile = "./sample_data/sample-results.txt";
double alpha = 0.00005;

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
    std::cout << "Running linear regressiong code iterative version" << std::endl;
    std::cout << "File with features data: " << featuresFile << std::endl;
    std::cout << "File with results data: " << resultsFile << std::endl;
    std::cout << "Starting with alpha: " << alpha << std::endl;
    readMatrixFromFile(featuresFile, x);
    readVectorFromFile(resultsFile, y);
    featureScale(x);

    linearRegressionCodeIterative(x, y, alpha);

    // linearRegressionCodeVectorized();

    // LOGISTIC REGRESSION

    // int m = 5;
    // int n = 2;
    // double alpha = 0.05;

    // featuresSet x;
    // featuresRow r;
    // std::vector<double> y;
    // std::vector<double> thetas;

    // for (int i = 0; i < n; ++i)
    // {
    //   thetas.push_back(0);
    // }

    // std::cout << "Thetas" << std::endl;
    // print(thetas);
    // std::cout << std::endl;
    // // Only possible values are 0 and 1
    // for (int i = 0; i < m; ++i)
    // {
    //   y.push_back(i % 2);
    // }

    // for (int i = 0; i < m; ++i)
    // {
    //   r.clear();
    //   for (int j = 0; j < n; ++j)
    //   {
    //     r.push_back(j + i);
    //   }
    //   r[0] = 1;
    //   x.push_back(r);
    // }

    // std::cout << "X" << std::endl;
    // for (int i = 0; i < m; ++i)
    // {
    //   print(x[i]);
    // }
    // std::cout << std::endl;

    // std::cout << "Y" << std::endl;
    // print(y);
    // std::cout << std::endl;

    // std::cout << "Thetas" << std::endl;
    // print(thetas);
    // std::cout << std::endl;

    // std::vector<double> costs = logisticRegressionCosts(x, y, thetas);
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
