#include "io.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

constexpr int STRING_WIDTH = 28;

void logger(const std::string& message) {
  std::cout << "# " << std::left << std::setw(STRING_WIDTH) << message
            << std::endl;
}

void logger(const std::string& message, const std::string& value) {
  std::cout << "# " << std::left << std::setw(STRING_WIDTH) << message
            << std::left << std::setw(STRING_WIDTH) << value << std::endl;
}

void logger(const std::string& message, double value) {
  std::cout << "# " << std::left << std::setw(STRING_WIDTH) << message
            << std::left << std::setw(STRING_WIDTH) << value << std::endl;
}

void logger(const std::string& message, std::complex<double> value) {
  std::cout << "# " << std::left << std::setw(STRING_WIDTH) << message
            << std::left << std::setw(STRING_WIDTH) << value << std::endl;
}

void logger(const std::string& message, double value1, double value2) {
  std::cout << "# " << std::left << std::setw(STRING_WIDTH) << message
            << std::left << std::setw(STRING_WIDTH) << value1 << std::left
            << std::setw(STRING_WIDTH) << value2 << std::endl;
}

void logger(const std::string& message, const std::vector<double>& value) {
  std::cout << "# " << std::left << std::setw(STRING_WIDTH) << message;
  for (int i = 0; i < static_cast<int>(value.size()); ++i) {
    std::cout << " " << value[i];
  }
  std::cout << std::endl;
}

std::string make_filename(const std::string& base_name, int config_number) {
  return base_name + "_" + std::to_string(config_number) + ".cnfg";
}

void read_gauge_field(field<gauge>& U, const std::string& base_name,
                      int config_number) {
  std::string filename = make_filename(base_name, config_number);
  std::ifstream input(filename.c_str(), std::ios::binary);
  if (input.good()) {
    double plaq_check;
    // read average plaquette as checksum
    input.read(reinterpret_cast<char*>(&plaq_check), sizeof(plaq_check));
    // read U
    input.read(reinterpret_cast<char*>(&(U[0][0](0, 0))),
               U.V * 4 * 9 * sizeof(std::complex<double>));
    // check that plaquette matches checksum
    double plaq = checksum_plaquette(U);
    if (fabs(plaq - plaq_check) > 1.e-13) {
      // if it doesn't match try the other
      // possible index ordering EO <-> LEXI
      lattice grid_tmp(U.L0, U.L1, U.L2, U.L3, !U.grid.isEO);
      field<gauge> U_tmp(grid_tmp);
      U_tmp = U;
      for (int ix = 0; ix < U.V; ++ix) {
        int ix_U;
        if (U.grid.isEO) {
          ix_U = U.grid.EO_from_LEXI(ix);
        } else {
          ix_U = U.grid.LEXI_from_EO(ix);
        }
        U[ix_U] = U_tmp[ix];
      }
      // check that plaquette matches checksum
      plaq = checksum_plaquette(U);
      if (fabs(plaq - plaq_check) > 1.e-13) {
        logger("ERROR: read_gauge_field CHECKSUM fail!");
        logger("filename: " + filename);
        logger("checksum plaquette in file", plaq_check);
        logger("measured plaquette", plaq);
        logger("deviation", plaq - plaq_check);
        exit(1);
      }
    }
    logger("Gauge field [" + filename + "] read with plaquette: ", plaq);
  } else {
    logger("Failed to read from file: " + filename);
    exit(1);
  }
}

void write_gauge_field(field<gauge>& U, const std::string& base_name,
                       int config_number) {
  std::string filename = make_filename(base_name, config_number);
  std::ofstream output(filename.c_str(), std::ios::binary);
  if (output.good()) {
    double plaq = checksum_plaquette(U);
    // write average plaquette as checksum
    output.write(reinterpret_cast<char*>(&plaq), sizeof(plaq));
    // write U
    output.write(reinterpret_cast<char*>(&(U[0][0](0, 0))),
                 U.V * 4 * 9 * sizeof(std::complex<double>));
    logger("Gauge field [" + filename + "] written with plaquette: ", plaq);
  } else {
    logger("Failed to write to file: " + filename);
  }
}

double checksum_plaquette(const field<gauge>& U) {
  double p = 0;
  //#pragma omp parallel for reduction (+:p)
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 1; mu < 4; ++mu) {
      for (int nu = 0; nu < mu; nu++) {
        p += ((U[ix][mu] * U.up(ix, mu)[nu]) *
              ((U[ix][nu] * U.up(ix, nu)[mu]).adjoint()))
                 .trace()
                 .real();
      }
    }
  }
  return p / static_cast<double>(3 * 6 * U.V);
}
