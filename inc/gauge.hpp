#ifndef LKEEGAN_CANONICAL_GAUGE_H
#define LKEEGAN_CANONICAL_GAUGE_H
#include "4d.hpp"
#include "su3.hpp"

std::complex<double> polyakov_loop(const field<gauge> &U, int mu);

#endif  // LKEEGAN_CANONICAL_GAUGE_H