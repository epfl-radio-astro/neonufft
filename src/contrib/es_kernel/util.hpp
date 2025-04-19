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

} // namespace contrib
} // namespace neonufft
