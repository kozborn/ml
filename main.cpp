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
#include "linear-regression.h"

int main()
{
  try
  {
    std::cout << "Machine learning - v.0.0.2 - classification" << std::endl;
    // linearRegressionCodeIterative();
    // linearRegressionCodeVectorized();

    // LOGISTIC REGRESSION

    int m = 5;
    int n = 2;

    featuresSet x;
    featuresRow r;
    std::vector<int> y;
    std::vector<double> thetas;

    for (int i = 0; i < n; ++i)
    {
      thetas.push_back(0);
    }

    std::cout << "Thetas" << std::endl;
    print(thetas);
    std::cout << std::endl;
    // Only possible values are 0 and 1
    for (int i = 0; i < m; ++i)
    {
      y.push_back(i % 2);
    }

    for (int i = 0; i < m; ++i)
    {
      r.clear();
      for (int j = 0; j < n; ++j)
      {
        r.push_back(j + i);
      }
      r[0] = 1;
      x.push_back(r);
    }

    std::cout << "X" << std::endl;
    for (int i = 0; i < m; ++i)
    {
      print(x[i]);
    }
    std::cout << std::endl;

    std::cout << "Y" << std::endl;
    print(y);
    std::cout << std::endl;

    std::cout << "Logistic cost fn: " << std::endl;
    std::cout << logisticCostFn(thetas, x, y);

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
