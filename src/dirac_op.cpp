#include "dirac_op.hpp"
#include <iostream>  //FOR DEBUGGING

dirac_op::dirac_op(const lattice& grid, double mass, double mu_I)
    : rng(123), mass(mass), mu_I(mu_I), eta(grid) {
  construct_eta(eta, grid);
}

void dirac_op::construct_eta(field<gamma_matrices>& eta, const lattice& grid) {
  for (int l = 0; l < grid.L3; ++l) {
    for (int k = 0; k < grid.L2; ++k) {
      for (int j = 0; j < grid.L1; ++j) {
        for (int i = 0; i < grid.L0; ++i) {
          int ix = grid.index(i, j, k, l);
          eta[ix][0] = +1.0;
          eta[ix][1] = +1.0 - 2.0 * (i % 2);
          eta[ix][2] = +1.0 - 2.0 * ((i + j) % 2);
          eta[ix][3] = +1.0 - 2.0 * ((i + j + k) % 2);
          eta[ix][4] = +1.0 - 2.0 * ((i + j + k + l) % 2);
        }
      }
    }
  }
}

void dirac_op::apbcs_in_time(field<gauge>& U) const {
  if (ANTI_PERIODIC_BCS) {
    for (int ix = 0; ix < U.VOL3; ++ix) {
      int i4x = U.it_ix(U.L0 - 1, ix);
      U[i4x][0] *= -1;
    }
  }
}

void dirac_op::apply_eta_bcs_to_U(field<gauge>& U) {
  if (!GAUGE_LINKS_INCLUDE_ETA_BCS) {
    for (int ix = 0; ix < U.V; ++ix) {
      for (int mu = 1; mu < 4; ++mu) {
        // eta[mu=0] = 1 so skip it
        U[ix][mu] *= eta[ix][mu];
      }
    }
    if (ANTI_PERIODIC_BCS) {
      for (int ix = 0; ix < U.VOL3; ++ix) {
        int i4x = U.it_ix(U.L0 - 1, ix);
        U[i4x][0] *= -1;
      }
    }
    GAUGE_LINKS_INCLUDE_ETA_BCS = true;
  }
}

void dirac_op::remove_eta_bcs_from_U(field<gauge>& U) {
  if (GAUGE_LINKS_INCLUDE_ETA_BCS) {
    for (int ix = 0; ix < U.V; ++ix) {
      for (int mu = 1; mu < 4; ++mu) {
        U[ix][mu] *= eta[ix][mu];
      }
    }
    if (ANTI_PERIODIC_BCS) {
      for (int ix = 0; ix < U.VOL3; ++ix) {
        int i4x = U.it_ix(U.L0 - 1, ix);
        U[i4x][0] *= -1;
      }
    }
    GAUGE_LINKS_INCLUDE_ETA_BCS = false;
  }
}

void dirac_op::gauge_fix_axial(field<gauge>& U) const {
  // determine gauge transform G
  field<gauge> G(U.grid);
  for (int ix = 0; ix < G.VOL3; ++ix) {
    // starting with G = 1 at T=0 boundary
    int ir = G.it_ix(0, ix);
    G[ir][0].setIdentity();
    // choose G to set all bulk timelike links to identity
    for (int it = 0; it < G.L0 - 1; ++it) {
      int il = ir;
      ir = G.iup(il, 0);
      G[ir][0] = G[il][0] * U[il][0];
    }
  }
  // apply gauge transform G to gauge links U
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      U[ix][mu] = G[ix][0] * U[ix][mu] * G.up(ix, mu)[0].adjoint();
    }
  }
}

void dirac_op::gaussian_P(field<gauge>& P) {
  // normal distribution p(x) ~ exp(-x^2/2), i.e. mu=0, sigma=1:
  // NO openMP: rng not threadsafe!
  std::normal_distribution<double> randdist_gaussian(0, 1.0);
  SU3_Generators T;
  for (int ix = 0; ix < P.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      P[ix][mu].setZero();
      for (int i = 0; i < 8; ++i) {
        std::complex<double> c(randdist_gaussian(rng), 0.0);
        P[ix][mu] += c * T[i];
      }
    }
  }
}

void dirac_op::random_U(field<gauge>& U, double eps) {
  // NO openMP: rng not threadsafe!
  gaussian_P(U);
  if (eps > 1.0) {
    eps = 1.0;
  }
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      U[ix][mu] = exp_ch((std::complex<double>(0.0, eps) * U[ix][mu]));
    }
  }
}

// explicitly construct dirac op as dense (3xVOL)x(3xVOL) matrix
// using normalisation D = 2m + U..
Eigen::MatrixXcd dirac_op::D_dense_matrix(field<gauge>& U) {
  Eigen::MatrixXcd D_matrix =
      Eigen::MatrixXcd::Zero(N_gauge * U.V, N_gauge * U.V);
  apply_eta_bcs_to_U(U);
  std::complex<double> mu_I_plus_factor = std::polar(1.0, mu_I);
  std::complex<double> mu_I_minus_factor = std::polar(1.0, -mu_I);
  for (int ix = 0; ix < U.V; ++ix) {
    int m_ix = N_gauge * ix;
    D_matrix.block<N_gauge, N_gauge>(m_ix, m_ix) =
        2.0 * mass * SU3mat::Identity();
    // mu=0 terms have extra chemical potential isospin factors
    // exp(+- i \mu_I): NB eta[ix][0] is just 1 so dropped from this expression
    int m_ix_up = N_gauge * U.iup(ix, 0);
    int m_ix_dn = N_gauge * U.idn(ix, 0);
    D_matrix.block<N_gauge, N_gauge>(m_ix, m_ix_up) =
        mu_I_plus_factor * U[ix][0];
    D_matrix.block<N_gauge, N_gauge>(m_ix, m_ix_dn) =
        -mu_I_minus_factor * U.dn(ix, 0)[0].adjoint().eval();
    for (int mu = 1; mu < 4; ++mu) {
      int m_ix_up = N_gauge * U.iup(ix, mu);
      int m_ix_dn = N_gauge * U.idn(ix, mu);
      D_matrix.block<N_gauge, N_gauge>(m_ix, m_ix_up) = U[ix][mu];
      D_matrix.block<N_gauge, N_gauge>(m_ix, m_ix_dn) =
          -U.dn(ix, mu)[mu].adjoint().eval();
    }
  }
  remove_eta_bcs_from_U(U);
  return D_matrix;
}

Eigen::MatrixXcd dirac_op::D_eigenvalues(field<gauge>& U) {
  // construct explicit dense dirac matrix
  Eigen::MatrixXcd D_matrix = D_dense_matrix(U);
  // find all eigenvalues of non-hermitian dirac operator matrix D
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
  ces.compute(D_matrix, false);
  return ces.eigenvalues();
}

Eigen::MatrixXcd dirac_op::DDdagger_eigenvalues(field<gauge>& U) {
  // construct explicit dense dirac matrix
  Eigen::MatrixXcd D_matrix = D_dense_matrix(U);
  D_matrix = D_matrix * D_matrix.adjoint();
  // find all eigenvalues of hermitian dirac operator matrix DDdagger
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
  saes.compute(D_matrix);
  return saes.eigenvalues();
}

Eigen::MatrixXcd dirac_op::B_dense_matrix(field<gauge>& U, int it) {
  Eigen::MatrixXcd B_matrix =
      Eigen::MatrixXcd::Zero(2 * N_gauge * U.VOL3, 2 * N_gauge * U.VOL3);
  apply_eta_bcs_to_U(U);
  // top-left corner matrix B(it)
  for (int ix3 = 0; ix3 < U.VOL3; ++ix3) {
    int m_ix = N_gauge * ix3;
    B_matrix.block<N_gauge, N_gauge>(m_ix, m_ix) =
        2.0 * mass * SU3mat::Identity();
    int ix4 = U.it_ix(it, ix3);
    for (int mu = 1; mu < 4; ++mu) {
      int m_ix_up = N_gauge * (U.iup(ix4, mu) - it) / U.L0;
      int m_ix_dn = N_gauge * (U.idn(ix4, mu) - it) / U.L0;
      B_matrix.block<N_gauge, N_gauge>(m_ix, m_ix_up) = U[ix4][mu];
      B_matrix.block<N_gauge, N_gauge>(m_ix, m_ix_dn) =
          -U.dn(ix4, mu)[mu].adjoint().eval();
    }
  }
  remove_eta_bcs_from_U(U);
  // top-right and bottom-left identity matrices
  for (int m_ix = 0; m_ix < N_gauge * U.VOL3; ++m_ix) {
    B_matrix(m_ix, m_ix + N_gauge * U.VOL3) = 1.0;
    B_matrix(m_ix + N_gauge * U.VOL3, m_ix) = 1.0;
  }
  return B_matrix;
}

Eigen::MatrixXcd dirac_op::P_eigenvalues(field<gauge>& U, double theta) {
  Eigen::MatrixXcd P_matrix =
      Eigen::MatrixXcd::Zero(2 * N_gauge * U.VOL3, 2 * N_gauge * U.VOL3);
  // gauge fix U to axial gauge
  gauge_fix_axial(U);
  // multiply single gauge link at (T-1,ix3=0) by e^{i 2pi theta}
  int ix4 = U.it_ix(U.L0 - 1, 0);
  U[ix4][0] = U[ix4][0] * std::polar<double>(1.0, 6.28318530718 * theta);
  // construct
  // ((U, 0), (0, U))
  // where U is the set of L^3 timelike links on timeslice T-1
  for (int ix3 = 0; ix3 < U.VOL3; ++ix3) {
    int m_ix = N_gauge * ix3;
    int ix4 = U.it_ix(U.L0 - 1, ix3);
    P_matrix.block<N_gauge, N_gauge>(m_ix, m_ix) = U[ix4][0];
    P_matrix.block<N_gauge, N_gauge>(m_ix + N_gauge * U.VOL3,
                                     m_ix + N_gauge * U.VOL3) = U[ix4][0];
  }
  for (int it = U.L0 - 1; it >= 0; --it) {
    P_matrix = B_dense_matrix(U, it) * P_matrix;
  }
  // find all eigenvalues of non-hermitian matrix P
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
  ces.compute(P_matrix, false);
  return ces.eigenvalues();
}