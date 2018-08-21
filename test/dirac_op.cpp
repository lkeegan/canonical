#include "dirac_op.hpp"
#include <iostream>
#include "catch.hpp"
#include "io.hpp"

constexpr double EPS = 5.e-14;

TEST_CASE("Axial gauge fixing sets U_0 to 1 except at T-1 boundary",
          "[gauge-fix]") {
  lattice grid(4);
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
  lattice grid(4);
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