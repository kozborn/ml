#ifndef UTILS_LIB
#define UTILS_LIB

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

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