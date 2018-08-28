#include <iomanip>
#include <iostream>
#include <random>
#include "gauge.hpp"
#include "io.hpp"

int main(int argc, char* argv[]) {
  if (argc - 1 != 1) {
    std::cout << "Gauge config base name not specified, e.g." << std::endl;
    std::cout << "./evals beta5.04_mass0.05" << std::endl;
    return 1;
  }

  std::string base_name(argv[1]);
  lattice grid(4, 6);

  std::cout.precision(14);
  logger("Polyakov loop");
  logger("T", grid.L0);
  logger("L", grid.L1);

  field<gauge> U(grid);
  for (int i_config = 1;; ++i_config) {
    read_gauge_field(U, base_name, i_config);

    std::complex<double> p = polyakov_loop(U, 0);
    std::cout << i_config << "\tT\t" << p.real() << "\t" << p.imag()
              << std::endl;
    p = polyakov_loop(U, 1) + polyakov_loop(U, 2) + polyakov_loop(U, 3);
    p /= 3.0;
    std::cout << i_config << "\tL\t" << p.real() << "\t" << p.imag()
              << std::endl;
  }
  return 0;
}