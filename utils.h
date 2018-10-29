#ifndef UTILS_LIB
#define UTILS_LIB

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

typedef std::vector<double> resultsSet;
typedef std::vector<double> featuresRow;
typedef std::vector<featuresRow> featuresSet;

void loadMatrix(std::istream &is, std::vector<std::vector<double>> &matrix, const std::string &delimeter)
{
  std::string line;
  std::string strnum;

  matrix.clear();

  while (std::getline(is, line))
  {
    matrix.push_back(std::vector<double>());
    for (std::string::const_iterator i = line.begin(); i != line.end(); i++)
    {
      if (delimeter.find(*i) == std::string::npos)
      {
        strnum += *i;
        if (i + 1 != line.end())
          continue;
      }
      if (strnum.empty())
        continue;

      double number;
      std::istringstream(strnum) >> number;
      matrix.back().push_back(number);
      strnum.clear();
    }
  }
}

void readMatrixFromFile(const std::string filename, featuresSet &X)
{
  std::cout << "Reading matrix from file: " << filename << std::endl;
  std::ifstream inputFile;

  inputFile.open(filename);
  if (inputFile.good())
  {
    loadMatrix(inputFile, X, " ");
  }
  else
  {
    std::cerr << "Cannot open file" << filename << std::endl;
  }
}

void readVectorFromFile(const std::string filename, resultsSet &Y)
{
  std::cout << "Reading vector from file: " << filename << std::endl;
  std::ifstream inputFile;
  double y = 0.0;
  inputFile.open(filename);
  if (inputFile.good())
  {
    while (inputFile >> y)
    {
      Y.push_back(y);
    }
  }
  else
  {
    std::cerr << "Cannot open file" << filename << std::endl;
  }
}

void append1toFeaturesSet(featuresSet &x)
{
  // Inserting 1 at the first column of training set
  for (int i = 0; i < x.size(); ++i)
  {
    x[i].insert(x[i].begin(), 1);
  }
}

int random(int min, int max)
{
  return std::rand() % (max - min + 1) + min;
}

inline bool isEqual(double x, double y)
{
  const double epsilon = 0.00000001;
  return fabs(x - y) < epsilon;
}

template <typename T>
void print(const std::vector<T> &v)
{
  std::streamsize ss = std::cout.precision();
  std::cout.precision(4);
  std::cout << std::fixed;
  std::cout << "[";
  for (int i = 0; i < v.size(); ++i)
  {
    std::cout << v[i];
    if (i < (v.size() - 1))
      std::cout << ", ";
  }
  std::cout << "]";
  std::cout.precision(ss);
  std::cout << std::endl;
}

template <typename T>
std::string vectorToString(const std::vector<T> &v)
{
  std::stringstream ss;

  ss << "[";
  for (int i = 0; i < v.size(); ++i)
  {
    ss << v[i];
    if (i < (v.size() - 1))
      ss << ", ";
  }
  ss << "]";
  return ss.str();
}

template <typename T>
void fillVector(std::vector<T> &v, int size)
{
  for (int i = 0; i < size; ++i)
  {
    v.push_back(i);
  }
}

template <typename T>
void fillVector(std::vector<T> &v, int size, int min, int max)
{
  for (int i = 0; i < size; ++i)
  {
    v.push_back(random(min, max));
  }
}

template <typename T>
std::string vectorToOctaveFormat(std::string name, const std::vector<T> &v)
{
  std::stringstream ss;
  ss << "# name: " << name << std::endl;
  ss << "# type: matrix" << std::endl;
  ss << "# rows: 1" << std::endl;
  ss << "# columns: " << v.size() << std::endl;
  for (auto i : v)
  {
    ss << " " << i;
  }
  ss << std::endl;
  return ss.str();
}

#endif