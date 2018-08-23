#include "dirac_op.hpp"
#include <iostream>
#include "catch.hpp"
#include "io.hpp"

constexpr double EPS = 1.e-13;
constexpr double NORM =
    2.55;  // arbitrary normalisation factor for every eigenvalue to prevent
           // Determinants from overflowing

TEST_CASE("Axial gauge fixing sets U_0 to 1 except at T-1 boundary",
          "[gauge-fix]") {
  lattice grid(2);
  field<gauge> U(grid);
  dirac_op D(grid);
  D.random_U(U, 0.3);
  D.gauge_fix_axial(U);
  double dev = 0;
  for (int ix = 0; ix < U.VOL3; ++ix) {
    for (int it = 0; it < U.L0 - 1; ++it) {
      int i4x = U.it_ix(it, ix);
      dev += (U[i4x][0] - SU3mat::Identity()).norm();
    }
  }
  CAPTURE(dev);
  REQUIRE(dev < sqrt(U.V) * EPS);
}

TEST_CASE("Axial gauge fixing doesn't change plaq", "[gauge-fix]") {
  lattice grid(2);
  field<gauge> U(grid);
  dirac_op D(grid);
  D.random_U(U, 0.1);
  double old_plaq = checksum_plaquette(U);
  CAPTURE(old_plaq);
  D.gauge_fix_axial(U);
  double new_plaq = checksum_plaquette(U);
  CAPTURE(new_plaq);
  REQUIRE(old_plaq - new_plaq < EPS);
}

/*
TEST_CASE("Axial gauge fixing doesn't change massless Dirac op eigenvalues",
          "[gauge-fix]") {
  lattice grid(4);
  field<gauge> U(grid);
  dirac_op D(grid, 0.02, 0.107);
  D.random_U(U, 0.20);
  Eigen::MatrixXcd old_evals = D.D_eigenvalues(U);
  double old_det_re = 1.0, old_det_phase = 0.0;
  for (int i = 0; i < old_evals.rows(); ++i) {
    old_det_re *= std::abs(old_evals(i)) / NORM;  // normalise to avoid overflow
    old_det_phase += std::arg(old_evals(i));
  }
  //  CAPTURE(old_evals);
  CAPTURE(old_det_re);
  CAPTURE(old_det_phase);
  D.gauge_fix_axial(U);
  Eigen::MatrixXcd new_evals = D.D_eigenvalues(U);
  double new_det_re = 1.0, new_det_phase = 0.0;
  for (int i = 0; i < new_evals.rows(); ++i) {
    new_det_re *= std::abs(new_evals(i)) / NORM;  // normalise to avoid overflow
    new_det_phase += std::arg(new_evals(i));
  }
  // CAPTURE(new_evals);
  CAPTURE(new_det_re);
  CAPTURE(new_det_phase);
  REQUIRE(std::abs((new_det_re - old_det_re) / new_det_re) < 1e-10);
  REQUIRE(std::abs(new_det_phase - old_det_phase) < EPS);
}
*/

TEST_CASE("Dirac op determinant from reduced P matrix eigenvalues", "[det]") {
  lattice grid(4);
  field<gauge> U(grid);
  dirac_op D(grid, 0.033, 0.205);
  D.random_U(U, 0.19);
  D.gauge_fix_axial(U);
  // find Dirac op determinant directly
  Eigen::MatrixXcd D_evals = D.D_eigenvalues(U);
  double D_det_re = 1.0, D_det_phase = 0.0;
  for (int i = 0; i < D_evals.rows(); ++i) {
    D_det_re *= std::abs(D_evals(i)) / NORM;  // normalise to avoid overflow
    D_det_phase += std::arg(D_evals(i));
  }
  CAPTURE(D_evals.rows());
  //  CAPTURE(D_evals);
  CAPTURE(D_det_re);
  CAPTURE(D_det_phase);
  // calculate Dirac op det in terms of
  // eigenvalues of reduced matrix P
  Eigen::MatrixXcd P_evals = D.P_eigenvalues(U);
  double P_det_re = 1.0, P_det_phase = 0.0;
  for (int i = 0; i < P_evals.rows(); ++i) {
    std::complex<double> P_eval =
        (P_evals(i) +
         exp(-std::complex<double>(0.0, static_cast<double>(U.L0) * D.mu_I)));
    P_det_re *= std::abs(P_eval) / NORM /
                NORM;  // normalise (half as many evals as D so do it twice such
                       // that overall normalisation matches)
    P_det_phase += std::arg(P_eval);
  }
  CAPTURE(P_det_phase);
  P_det_phase += static_cast<double>(3 * U.V) * D.mu_I;
  CAPTURE(P_evals.rows());
  // CAPTURE(P_evals);
  CAPTURE(P_det_re);
  CAPTURE(P_det_phase);
  REQUIRE(std::abs((D_det_phase - P_det_phase)) < 1e-10);
  REQUIRE(std::abs((D_det_re - P_det_re) / D_det_re) < 1e-10);
}
