#include <iomanip>
#include <iostream>
#include <random>
#include "dirac_op.hpp"
#include "io.hpp"

int main(int argc, char *argv[]) {
  if (argc - 1 != 0) {
    std::cout << "Gauge field not specified, e.g." << std::endl;
    std::cout << "./evals 4_4_b6.0.cnfg" << std::endl;
    return 1;
  }

  // Use mpfr++ multiple precision reals
  // Set bits of precision for type mpfr::mpreal
  constexpr int BITS_OF_PRECISION = 1024;
  mpfr::mpreal::set_default_prec(BITS_OF_PRECISION);

  double mass = 0.05;

  std::cout.precision(14);
  logger("Fourier coefficients");
  logger("Bits of precision:", BITS_OF_PRECISION);
  logger("mass:", mass);

  lattice grid(4);
  field<gauge> U(grid);
  dirac_op D(grid, mass);
  D.random_U(U, 0.19);

  // Get eigenvalues of reduced matrix P
  Eigen::MatrixXcd P_evals = D.P_eigenvalues(U);

  // Do recursion in extended precision
  // NOTE: for error checking of recursion, can reorder eigenvalues, or add
  // random 1e-15 values, and see how both of these affect the output of
  // extended precision calculations below
  int k_max = 6 * U.VOL3;
  std::vector<mpfr::mpreal> c_old_re(k_max + 1, 0.0);
  std::vector<mpfr::mpreal> c_old_im(k_max + 1, 0.0);
  std::vector<mpfr::mpreal> c_new_re(k_max + 1, 0.0);
  std::vector<mpfr::mpreal> c_new_im(k_max + 1, 0.0);
  c_old_re[0] = 1.0;
  for (int n = 0; n < k_max; ++n) {
    mpfr::mpreal e_re = P_evals(n).real();
    mpfr::mpreal e_im = P_evals(n).imag();
    c_new_re[0] = e_re * c_old_re[0] - e_im * c_old_im[0];
    c_new_im[0] = e_im * c_old_re[0] + e_re * c_old_im[0];
    for (int k = 1; k <= n + 1; ++k) {
      c_new_re[k] = e_re * c_old_re[k] - e_im * c_old_im[k] + c_old_re[k - 1];
      c_new_im[k] = e_re * c_old_im[k] + e_im * c_old_re[k] + c_old_im[k - 1];
    }
    c_old_re = c_new_re;
    c_old_im = c_new_im;
  }

  // Q = k - k_max/2
  std::cout << "# Coeffs (Q, ln|magnitude|, phase): " << std::endl;
  for (int k = 0; k <= k_max; ++k) {
    int Q = k - k_max / 2;
    mpfr::mpreal magnitude =
        c_new_re[k] * c_new_re[k] + c_new_im[k] * c_new_im[k];
    std::complex<double> c_normalised(
        static_cast<double>(c_new_re[k] / magnitude),
        static_cast<double>(c_new_im[k] / magnitude));
    double phase = std::arg(c_normalised);

    std::cout << Q << "\t" << log(magnitude) << "\t" << phase << std::endl;
  }

  return 0;
}