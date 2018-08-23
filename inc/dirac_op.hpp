#ifndef LKEEGAN_MURHMC_DIRAC_OP_H
#define LKEEGAN_MURHMC_DIRAC_OP_H
#include <random>
#include "4d.hpp"
#include "Eigen3/Eigen/Eigenvalues"
#include "omp.h"
#include "su3.hpp"

// staggered space-dependent gamma matrices
// for now stored as 5x doubles per site but they are just +/- signs, and g[0]
// is just + everywhere g[4] is gamma_5
class gamma_matrices {
 private:
  double g_[5];

 public:
  double& operator[](int i) { return g_[i]; }
  double operator[](int i) const { return g_[i]; }
};

// Staggered dirac operator
class dirac_op {
 private:
  // Construct staggered eta (gamma) matrices
  void construct_eta(field<gamma_matrices>& eta, const lattice& grid);

 public:
  std::ranlux48 rng;
  double mass;
  double mu_I;
  field<gamma_matrices> eta;
  bool ANTI_PERIODIC_BCS = true;
  bool GAUGE_LINKS_INCLUDE_ETA_BCS = false;

  dirac_op(const lattice& grid, double mass, double mu_I = 0.0);

  explicit dirac_op(const lattice& grid) : dirac_op::dirac_op(grid, 0.0, 0.0) {}

  void apbcs_in_time(field<gauge>& U) const;

  // Applies eta matrices and apbcs in time to the gauge links U
  // Required before and after using EO versions of dirac op
  // Toggles flag GAUGE_LINKS_INCLUDE_ETA_BCS
  void apply_eta_bcs_to_U(field<gauge>& U);
  void remove_eta_bcs_from_U(field<gauge>& U);

  // Axial gauge: all timelike links 1 except at T-1 boundary
  void gauge_fix_axial(field<gauge>& U) const;

  void gaussian_P(field<gauge>& P);
  void random_U(field<gauge>& U, double eps);

  // Returns eigenvalues of Dirac op
  // Explicitly constructs dense (3*VOL)x(3*VOL) matrix Dirac op and finds all
  // eigenvalues
  Eigen::MatrixXcd D_eigenvalues(field<gauge>& U);

  // Same for DDdagger, but much faster since we can use a hermitian solver.
  Eigen::MatrixXcd DDdagger_eigenvalues(field<gauge>& U);

  // explicitly construct dirac op as dense (3*VOL)x(3*VOL) matrix
  Eigen::MatrixXcd D_dense_matrix(field<gauge>& U);

  // explicitly construct dense (2x3xVOL3)x(2x3xVOL3) matrix P
  // diagonalise and return all eigenvalues
  // NOTE: also gauge fixes U to axial gauge
  Eigen::MatrixXcd P_eigenvalues(field<gauge>& U);

  // explicitly construct dense (2x3xVOL3)x(2x3xVOL3) matrix
  // B at timeslice it, using normalisation D = 2m + U..
  // MUST be lexi grid layout for U!
  Eigen::MatrixXcd B_dense_matrix(field<gauge>& U, int it);
};

#endif  // LKEEGAN_MURHMC_DIRAC_OP_H