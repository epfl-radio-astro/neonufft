#pragma once

/*
  Taken from:
    https://people.sc.fsu.edu/~jburkardt/cpp_src/legendre_rule_fast/legendre_rule_fast.html

  Licensing:

    This code is distributed under the MIT license.


  Modified for GPU usage.
*/

#include "neonufft/config.h"
//---

#include "neonufft/gpu/util/runtime.hpp"
#include "neonufft/gpu/util/runtime_api.hpp"
#include "neonufft/util/math.hpp"

namespace neonufft {
namespace gpu {

template<typename T>
__device__ static T ts_mult(T* u, T h, int n)

//****************************************************************************80
//
//  Purpose:
//
//    TS_MULT evaluates a polynomial.
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    17 May 2013
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Parameters:
//
//    Input, T U[N+1], the polynomial coefficients.
//    U[0] is ignored.
//
//    Input, T H, the polynomial argument.
//
//    Input, int N, the number of terms to compute.
//
//    Output, T TS_MULT, the value of the polynomial.
//
{
  T hk;
  int k;
  T ts;

  ts = T(0);
  hk = T(1);
  for (k = 1; k <= n; k++) {
    ts = ts + u[k] * hk;
    hk = hk * h;
  }
  return ts;
}

template<typename T>
__device__ static T rk2_leg(T t1, T t2, T x, int n)

//****************************************************************************80
//
//  Purpose:
//
//    RK2_LEG advances the value of X(T) using a Runge-Kutta method.
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    22 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Parameters:
//
//    Input, T T1, T2, the range of the integration interval.
//
//    Input, T X, the value of X at T1.
//
//    Input, int N, the number of steps to take.
//
//    Output, T RK2_LEG, the value of X at T2.
//
{
  T f;
  T h;
  int j;
  T k1;
  T k2;
  int m = 10;
  T snn1;
  T t;

  h = (t2 - t1) / (T)m;
  snn1 = std::sqrt((T)(n * (n + 1)));
  t = t1;

  for (j = 0; j < m; j++) {
    f = (T(1) - x) * (T(1) + x);
    k1 = -h * f / (snn1 * std::sqrt(f) - T(0.5) * x * std::sin(T(2) * t));
    x = x + k1;

    t = t + h;

    f = (T(1) - x) * (T(1) + x);
    k2 = -h * f / (snn1 * std::sqrt(f) - T(0.5) * x * std::sin(T(2) * t));
    x = x + T(0.5) * (k2 - k1);
  }
  return x;
}

//****************************************************************************80

template<typename T>
__device__ static void legendre_compute_glr0(int n, T* p, T* pp)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR0 gets a starting value for the fast algorithm.
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    19 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, int N, the order of the Legendre polynomial.
//
//    Output, T *P, *PP, the value of the N-th Legendre polynomial
//    and its derivative at 0.
//
{
  T dk;
  int k;
  T pm1;
  T pm2;
  T ppm1;
  T ppm2;

  pm2 = T(0);
  pm1 = T(1);
  ppm2 = T(0);
  ppm1 = T(0);

  for (k = 0; k < n; k++) {
    dk = (T)k;
    *p = -dk * pm2 / (dk + T(1));
    *pp = ((T(2) * dk + T(1)) * pm1 - dk * ppm2) / (dk + T(1));
    pm2 = pm1;
    pm1 = *p;
    ppm2 = ppm1;
    ppm1 = *pp;
  }
  return;
}
//****************************************************************************80

template<typename T>
__device__ static void legendre_compute_glr1(int n, T* __restrict__ x, T* __restrict__ w)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR1 gets the complete set of Legendre points and weights.
//
//  Discussion:
//
//    This routine requires that a starting estimate be provided for one
//    root and its derivative.  This information will be stored in entry
//    (N+1)/2 if N is odd, or N/2 if N is even, of X and W.
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    19 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, int N, the order of the Legendre polynomial.
//
//    Input/output, T X[N].  On input, a starting value
//    has been set in one entry.  On output, the roots of the Legendre
//    polynomial.
//
//    Input/output, T W[N].  On input, a starting value
//    has been set in one entry.  On output, the derivatives of the Legendre
//    polynomial at the zeros.
//
//  Local Parameters:
//
//    Local, int M, the number of terms in the Taylor expansion.
//
{
  T dk;
  T dn;
  T h;
  int j;
  int k;
  int l;
  int n2;
  int s;
  T xp;

  if (n % 2 == 1) {
    n2 = (n - 1) / 2 - 1;
    s = 1;
  } else {
    n2 = n / 2 - 1;
    s = 0;
  }

  constexpr int m = 30;
  T u[m + 2];
  T up[m + 1];

  dn = (T)n;

  for (j = n2 + 1; j < n - 1; j++) {
    xp = x[j];

    h = rk2_leg(math::pi<T> / T(2), -math::pi<T> / T(2), xp, n) - xp;

    u[0] = T(0);
    u[1] = T(0);
    u[2] = w[j];

    up[0] = T(0);
    up[1] = u[2];

    for (k = 0; k <= m - 2; k++) {
      dk = (T)k;

      u[k + 3] = (T(2) * xp * (dk + T(1)) * u[k + 2] +
                  (dk * (dk + T(1)) - dn * (dn + T(1))) * u[k + 1] / (dk + T(1))) /
                 (T(1) - xp) / (T(1) + xp) / (dk + T(2));

      up[k + 2] = (dk + T(2)) * u[k + 3];
    }

    for (l = 0; l < 5; l++) {
      h = h - ts_mult(u, h, m) / ts_mult(up, h, m - 1);
    }

    x[j + 1] = xp + h;
    w[j + 1] = ts_mult(up, h, m - 1);
  }

  for (k = 0; k <= n2 + s; k++) {
    x[k] = -x[n - 1 - k];
    w[k] = w[n - 1 - k];
  }

  return;
}
//****************************************************************************80

template<typename T>
__device__ static void legendre_compute_glr2(T pn0, int n, T* __restrict__ x1,
                                             T* __restrict__ d1)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR2 finds the first real root.
//
//  Discussion:
//
//    This function is only called if N is even.
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    19 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, T PN0, the value of the N-th Legendre polynomial
//    at 0.
//
//    Input, int N, the order of the Legendre polynomial.
//
//    Output, T *X1, the first real root.
//
//    Output, T *D1, the derivative at X1.
//
//  Local Parameters:
//
//    Local, int M, the number of terms in the Taylor expansion.
//
{
  T dk;
  T dn;
  int k;
  int l;
  T t;

  constexpr int m = 30;
  T u[m + 2];
  T up[m + 1];

  t = T(0);
  *x1 = rk2_leg(t, -math::pi<T> / T(2), T(0), n);

  dn = (T)n;
  //
  //  U[0] and UP[0] are never used.
  //  U[M+1] is set, but not used, and UP[M] is set and not used.
  //  What gives?
  //
  u[0] = T(0);
  u[1] = pn0;

  up[0] = T(0);

  for (k = 0; k <= m - 2; k = k + 2) {
    dk = (T)k;

    u[k + 2] = T(0);
    u[k + 3] = (dk * (dk + T(1)) - dn * (dn + T(1))) * u[k + 1] / (dk + T(1)) / (dk + T(2));

    up[k + 1] = T(0);
    up[k + 2] = (dk + T(2)) * u[k + 3];
  }

  for (l = 0; l < 5; l++) {
    *x1 = *x1 - ts_mult(u, *x1, m) / ts_mult(up, *x1, m - 1);
  }
  *d1 = ts_mult(up, *x1, m - 1);

  return;
}
//****************************************************************************80

template<typename T>
__device__ static void legendre_compute_glr(int n, T* __restrict__ x, T* __restrict__ w)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR: Legendre quadrature by the Glaser-Liu-Rokhlin method.
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    20 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, int N, the order.
//
//    Output, T X[N], the abscissas.
//
//    Output, T W[N], the weights.
//
{
  if (threadIdx.x == 0) {
    T p;
    T pp;

    //
    //  Get the value and derivative of the N-th Legendre polynomial at 0.
    //
    legendre_compute_glr0(n, &p, &pp);

    if (n % 2 == 1) {
      //
      //  If N is odd, then zero is a root.
      //
      x[(n - 1) / 2] = p;
      w[(n - 1) / 2] = pp;
    } else {
      //
      //  If N is even, we have to call a function to find the first root.
      //
      legendre_compute_glr2(p, n, &x[n / 2], &w[n / 2]);
    }
    //
    //  Get the complete set of roots and derivatives.
    //
    legendre_compute_glr1(n, x, w);
    //
    //  Compute the W.
    //
  }
  __syncthreads();

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    w[i] = T(2) / (T(1) - x[i]) / (T(1) + x[i]) / w[i] / w[i];
  }

  __syncthreads();

  T w_sum = T(0);
  for (int i = 0; i < n; i++) {
    w_sum = w_sum + w[i];
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    w[i] = T(2) * w[i] / w_sum;
  }
  __syncthreads();
}
}  // namespace gpu
}  // namespace neonufft
