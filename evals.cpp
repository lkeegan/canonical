#include <iomanip>
#include <iostream>
#include <random>
#include "dirac_op.hpp"
#include "io.hpp"

int main(int argc, char *argv[]) {
  if (argc - 1 != 1) {
    std::cout << "Input file not specified, e.g." << std::endl;
    std::cout << "./evals input_file.txt" << std::endl;
    return 1;
  }

  std::cout.precision(14);

  logger("");

  return 0;
}