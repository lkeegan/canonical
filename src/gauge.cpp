#include "gauge.hpp"

std::complex<double> polyakov_loop(const field<gauge> &U, int mu) {
  std::complex<double> p = 0;
  int L[4] = {U.L0, U.L1, U.L2, U.L3};
  for (int ix = 0; ix < U.V; ++ix) {
    SU3mat P = U[ix][mu];
    int ixmu = ix;
    for (int n_mu = 1; n_mu < L[mu]; n_mu++) {
      ixmu = U.iup(ixmu, mu);
      P *= U[ixmu][mu];
    }
    p += P.trace();
  }
  return p / static_cast<double>(3 * U.V);
}
