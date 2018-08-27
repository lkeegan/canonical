#include <iomanip>
#include <iostream>
#include <random>
#include "dirac_op.hpp"
#include "io.hpp"

int main(int argc, char* argv[]) {
  if (argc - 1 != 1) {
    std::cout << "Gauge config base name not specified, e.g." << std::endl;
    std::cout << "./evals beta5.04_mass0.05" << std::endl;
    return 1;
  }

  std::string base_name(argv[1]);

  // Use mpfr++ multiple precision reals
  // Set bits of precision for type mpfr::mpreal
  constexpr int BITS_OF_PRECISION = 4096;
  mpfr::mpreal::set_default_prec(BITS_OF_PRECISION);

  double mass = 0.05;
  lattice grid(4, 6);

  std::cout.precision(14);
  logger("Fourier coefficients");
  logger("Bits of precision:", BITS_OF_PRECISION);
  logger("T", grid.L0);
  logger("L", grid.L1);
  logger("mass:", mass);

  field<gauge> U(grid);
  dirac_op D(grid, mass);
  for (int i_config = 1;; ++i_config) {
    read_gauge_field(U, base_name, i_config);

    // Get eigenvalues of reduced matrix P (expensive part)
    Eigen::MatrixXcd P_evals = D.P_eigenvalues(U);

    // Output eigenvalues
    std::cout << "# Eigenvalues (lambda_re, lambda_im): " << std::endl;
    for (int k = 0; k < P_evals.rows(); ++k) {
      std::cout << "#EV " << i_config << "\t" << P_evals(k).real() << "\t"
                << P_evals(k).imag() << std::endl;
    }

    // Do recursion in extended precision
    // NOTE: for checking actual precision of recursion, can reorder
    // eigenvalues, or add random 1e-15 values, and see how both of these affect
    // the output of extended precision calculations below
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

    // det(M(\mu=0)) = \sum c_Q
    mpfr::mpreal det(0), det_im(0);
    for (int k = 0; k <= k_max; ++k) {
      det += c_new_re[k];
      det_im += c_new_im[k];
    }
    std::cout << "# Det M: " << det << std::endl;
    std::cout << "# phase(Det M): " << atan2(det_im, det) << std::endl;

    // Output ZC(Q) = c_Q/det(M) for each Q
    // Q = k - k_max/2
    std::cout << "# Coeffs (Q, ZC_re, ZC_im, phase): " << std::endl;
    for (int k = 0; k <= k_max; ++k) {
      int Q = k - k_max / 2;
      mpfr::mpreal phase = atan2(c_new_im[k], c_new_re[k]);
      std::cout << i_config << "\t" << Q << "\t" << c_new_re[k] / det << "\t"
                << c_new_im[k] / det << "\t" << phase << std::endl;
    }
  }
  return 0;
}