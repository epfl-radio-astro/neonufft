#pragma once

// Copyright (C) 2017-2023 The Simons Foundation, Inc. - All Rights Reserved.

// ------

// FINUFFT is licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance with the
// License.  You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ------
// Modifications include namespace, types, naming 

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <array>

#include "neonufft/config.h"
//---

#include "contrib/legendre_rule/legendre_rule_fast.hpp"
#include "neonufft/types.hpp"
#include "neonufft/util/math.hpp"

namespace neonufft {

namespace contrib {

inline IntType next235even(IntType n)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). Adapted from fortran in hellskitchen.  Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
{
  if (n <= 2)
    return 2;
  if (n % 2 == 1)
    n += 1;              // even
  IntType nplus = n - 2; // to cancel out the +=2 at start of loop
  IntType numdiv = 2;    // a dummy that is >1
  while (numdiv > 1) {
    nplus += 2; // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0)
      numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0)
      numdiv /= 3;
    while (numdiv % 5 == 0)
      numdiv /= 5;
  }
  return nplus;
}

template<typename T>
inline T evaluate_es_kernel(T x, T ES_halfwidth, T ES_beta, T ES_c)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg finufft/onedim_* 2/17/17
*/
{
  if (std::abs(x) >= (T)ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return std::exp((T)ES_beta * (sqrt((T)1.0 - (T)ES_c * x * x) - 1));
}

template <typename T>
IntType t3_grid_size(T output_half_extent, T input_half_extent, T upsampfac,
                     IntType nspread) {
  IntType nss = nspread + 1;

  T input_half_extent_safe = input_half_extent,
    output_half_extent_safe = output_half_extent; // may be tweaked locally
  if (input_half_extent == 0.0) // logic ensures XS>=1, handle
                                // input_half_extent=0 a/o output_half_extent=0
    if (output_half_extent == 0.0) {
      input_half_extent_safe = 1.0;
      output_half_extent_safe = 1.0;
    } else
      input_half_extent_safe = std::max(input_half_extent_safe, 1 / output_half_extent);
  else
    output_half_extent_safe = std::max(output_half_extent_safe, 1 / input_half_extent);
  IntType size = 2.0 * upsampfac * output_half_extent_safe *
                     input_half_extent_safe / math::pi<T> +
                 nss;
  if (size < 2 * nspread)
    size = 2 * nspread;
  size += size % 2; // ensure nf is even. Size for optimal fft performance only
                    // required for upsampled grid.
  return size;
}

// modified to separate grid size calculation
template <typename T>
void set_nhg_type3(T output_half_extent, T input_half_extent, T upsampfac,
                   IntType grid_size, T *h, T *gam)
{
  *h = 2 * math::pi<T> / grid_size; // upsampled grid spacing

  T output_half_extent_safe = output_half_extent; // may be tweaked locally
  if (input_half_extent == 0 && output_half_extent == 0) {
    output_half_extent_safe = 1.0;
  } else if (output_half_extent == 0) {
    output_half_extent_safe = std::max(output_half_extent_safe, 1 / input_half_extent);
  }

  *gam = T(grid_size) / (2.0 * upsampfac * output_half_extent_safe); // x scale fac to x'
}

template <typename T>
inline void onedim_fseries_kernel_inverse(IntType nf, T *fwkerhalf,
                                          IntType nspread, T ES_halfwidth,
                                          T ES_beta, T ES_c)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 Ts)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.
 */
{
  T J2 = nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 3.0 * J2); // not sure why so large? cannot exceed MAX_NQUAD
  std::vector<T> f(q);
  std::vector<double> z(2 * q);
  std::vector<double> w(2 * q);
  quadrature::legendre_compute_glr(
      2 * q, z.data(),
      w.data()); // only half the nodes used, eg on (0,1)
  std::vector<std::complex<T>> a(q);
  const IntType nfHalf = nf / 2;
  for (int n = 0; n < q; ++n) { // set up nodes z_n and vals f_n
    z[n] *= J2;                 // rescale nodes
    f[n] = J2 * (T)w[n] *
           evaluate_es_kernel((T)z[n], ES_halfwidth, ES_beta,
                              ES_c); // vals & quadr wei
    a[n] = std::exp(T(2 * math::pi<T>) * std::complex<T>{0, 1} * (T)(nfHalf - z[n]) /
               (T)nf); // phase winding rates
  }

  IntType nout = nf / 2 + 1; // how many values we're writing to

  std::vector<std::complex<T>> aj(q); // phase rotator for this thread
  for (int n = 0; n < q; ++n)
    aj[n] = 1;

  for (IntType j = 0; j < nout; ++j) { // loop along output array
    T x = 0.0;                         // accumulator for answer at this j
    for (int n = 0; n < q; ++n) {
      x += f[n] * 2 * real(aj[n]); // include the negative freq
      aj[n] *= a[n];               // wind the phases
    }
    fwkerhalf[j] = T(1) / x; // NOTE: modified to compute inverse
  }
}

} // namespace contrib
} // namespace neonufft
