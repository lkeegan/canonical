#ifndef LKEEGAN_CANONICAL_IO_H
#define LKEEGAN_CANONICAL_IO_H
#include <mpreal.h>
#include <complex>
#include <fstream>
#include <string>
#include "4d.hpp"
#include "su3.hpp"

// output message and variables to log file
void logger(const std::string& message);
void logger(const std::string& message, const std::string& value);
void logger(const std::string& message, double value);
void logger(const std::string& message, std::complex<double> value);
void logger(const std::string& message, double value1, double value2);
void logger(const std::string& message, const std::vector<double>& value);
void logger(const std::string& message,
            const std::vector<std::complex<double>>& value);
void logger(const std::string& message, std::vector<mpfr::mpreal>& re,
            std::vector<mpfr::mpreal>& im);

// returns a filename for config with given number
std::string make_fileName(const std::string& base_name, int config_number);

// read gauge field from file
void read_gauge_field(field<gauge>& U, const std::string& base_name,
                      int config_number);

// write gauge field to file
void write_gauge_field(field<gauge>& U, const std::string& base_name,
                       int config_number);

// calculate average plaqutte for use as checksum
double checksum_plaquette(const field<gauge>& U);

#endif  // LKEEGAN_CANONICAL_IO_H