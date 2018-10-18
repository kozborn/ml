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

int main()
{
  try
  {
    std::cout << "Squared cost function v.0.0.1" << std::endl;
    linearRegressionCodeIterative();
    linearRegressionCodeVectorized();

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
