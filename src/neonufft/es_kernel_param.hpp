#pragma once

#include "neonufft/config.h"

#include <cmath>
#include <algorithm>

#include "neonufft/util/math.hpp"

namespace neonufft {


template <typename T>
struct KernelParameters {
  T es_halfwidth = 0.0;
  T es_c = 0.0;
  T es_beta = 0.0;
  int n_spread = 0;
  double upsampfac = 2.0;
  bool approximation = true;

  KernelParameters() = default;

  KernelParameters(double tol, double upsamplefactor, bool kernel_approximation)
      : upsampfac(upsamplefactor), approximation(kernel_approximation) {
    n_spread = 2;
    if (upsamplefactor == 2.0) {
      n_spread = std::ceil(-std::log10((tol) / 10.0));
      // test. Sometimes doesn't match finufft due to numerics?
      // n_spread = std::ceil(-std::log10((opt_.tol + 1e-10) / 10.0));
    } else {
      n_spread = std::ceil(-std::log(tol) / (math::pi<double> *
                                             std::sqrt(1.0 - 1.0 / upsamplefactor)));
    }

    n_spread = std::max<int>(2, n_spread);
    n_spread = std::min<int>(16, n_spread);

    es_halfwidth = n_spread / 2.0;
    es_c = 4.0 / (n_spread * n_spread);

    // Factors from fiNUFFT
    double beta_fac = 2.30; 
    if (n_spread == 2)
      beta_fac = 2.20;
    if (n_spread == 3)
      beta_fac = 2.26;
    if (n_spread == 4)
      beta_fac = 2.38;
    if (upsamplefactor != 2.0) {
      // Eq. 4.5 in fiNUFFT paper
      const auto gamma = 0.97;
      beta_fac = gamma * math::pi<double> * (1.0 - 1.0 / (2 * upsamplefactor));
    }
    es_beta = beta_fac * n_spread;
  }
};

} // namespace neonufft
