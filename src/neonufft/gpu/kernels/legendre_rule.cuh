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

__device__ static double ts_mult(double* u, double h, int n)

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
//    Input, double U[N+1], the polynomial coefficients.
//    U[0] is ignored.
//
//    Input, double H, the polynomial argument.
//
//    Input, int N, the number of terms to compute.
//
//    Output, double TS_MULT, the value of the polynomial.
//
{
  double hk;
  int k;
  double ts;

  ts = 0.0;
  hk = 1.0;
  for (k = 1; k <= n; k++) {
    ts = ts + u[k] * hk;
    hk = hk * h;
  }
  return ts;
}

__device__ static double rk2_leg(double t1, double t2, double x, int n)

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
//    Input, double T1, T2, the range of the integration interval.
//
//    Input, double X, the value of X at T1.
//
//    Input, int N, the number of steps to take.
//
//    Output, double RK2_LEG, the value of X at T2.
//
{
  double f;
  double h;
  int j;
  double k1;
  double k2;
  int m = 10;
  double snn1;
  double t;

  h = (t2 - t1) / (double)m;
  snn1 = std::sqrt((double)(n * (n + 1)));
  t = t1;

  for (j = 0; j < m; j++) {
    f = (1.0 - x) * (1.0 + x);
    k1 = -h * f / (snn1 * std::sqrt(f) - 0.5 * x * std::sin(2.0 * t));
    x = x + k1;

    t = t + h;

    f = (1.0 - x) * (1.0 + x);
    k2 = -h * f / (snn1 * std::sqrt(f) - 0.5 * x * std::sin(2.0 * t));
    x = x + 0.5 * (k2 - k1);
  }
  return x;
}

//****************************************************************************80

__device__ static void legendre_compute_glr0(int n, double* p, double* pp)

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
//    Output, double *P, *PP, the value of the N-th Legendre polynomial
//    and its derivative at 0.
//
{
  double dk;
  int k;
  double pm1;
  double pm2;
  double ppm1;
  double ppm2;

  pm2 = 0.0;
  pm1 = 1.0;
  ppm2 = 0.0;
  ppm1 = 0.0;

  for (k = 0; k < n; k++) {
    dk = (double)k;
    *p = -dk * pm2 / (dk + 1.0);
    *pp = ((2.0 * dk + 1.0) * pm1 - dk * ppm2) / (dk + 1.0);
    pm2 = pm1;
    pm1 = *p;
    ppm2 = ppm1;
    ppm1 = *pp;
  }
  return;
}
//****************************************************************************80

__device__ static void legendre_compute_glr1(int n, double* __restrict__ x, double* __restrict__ w)

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
//    Input/output, double X[N].  On input, a starting value
//    has been set in one entry.  On output, the roots of the Legendre
//    polynomial.
//
//    Input/output, double W[N].  On input, a starting value
//    has been set in one entry.  On output, the derivatives of the Legendre
//    polynomial at the zeros.
//
//  Local Parameters:
//
//    Local, int M, the number of terms in the Taylor expansion.
//
{
  double dk;
  double dn;
  double h;
  int j;
  int k;
  int l;
  int n2;
  int s;
  double xp;

  if (n % 2 == 1) {
    n2 = (n - 1) / 2 - 1;
    s = 1;
  } else {
    n2 = n / 2 - 1;
    s = 0;
  }

  constexpr int m = 30;
  double u[m + 2];
  double up[m + 1];

  dn = (double)n;

  for (j = n2 + 1; j < n - 1; j++) {
    xp = x[j];

    h = rk2_leg(math::pi<double> / 2.0, -math::pi<double> / 2.0, xp, n) - xp;

    u[0] = 0.0;
    u[1] = 0.0;
    u[2] = w[j];

    up[0] = 0.0;
    up[1] = u[2];

    for (k = 0; k <= m - 2; k++) {
      dk = (double)k;

      u[k + 3] = (2.0 * xp * (dk + 1.0) * u[k + 2] +
                  (dk * (dk + 1.0) - dn * (dn + 1.0)) * u[k + 1] / (dk + 1.0)) /
                 (1.0 - xp) / (1.0 + xp) / (dk + 2.0);

      up[k + 2] = (dk + 2.0) * u[k + 3];
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

__device__ static void legendre_compute_glr2(double pn0, int n, double* __restrict__ x1,
                                             double* __restrict__ d1)

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
//    Input, double PN0, the value of the N-th Legendre polynomial
//    at 0.
//
//    Input, int N, the order of the Legendre polynomial.
//
//    Output, double *X1, the first real root.
//
//    Output, double *D1, the derivative at X1.
//
//  Local Parameters:
//
//    Local, int M, the number of terms in the Taylor expansion.
//
{
  double dk;
  double dn;
  int k;
  int l;
  double t;

  constexpr int m = 30;
  double u[m + 2];
  double up[m + 1];

  t = 0.0;
  *x1 = rk2_leg(t, -math::pi<double> / 2.0, 0.0, n);

  dn = (double)n;
  //
  //  U[0] and UP[0] are never used.
  //  U[M+1] is set, but not used, and UP[M] is set and not used.
  //  What gives?
  //
  u[0] = 0.0;
  u[1] = pn0;

  up[0] = 0.0;

  for (k = 0; k <= m - 2; k = k + 2) {
    dk = (double)k;

    u[k + 2] = 0.0;
    u[k + 3] = (dk * (dk + 1.0) - dn * (dn + 1.0)) * u[k + 1] / (dk + 1.0) / (dk + 2.0);

    up[k + 1] = 0.0;
    up[k + 2] = (dk + 2.0) * u[k + 3];
  }

  for (l = 0; l < 5; l++) {
    *x1 = *x1 - ts_mult(u, *x1, m) / ts_mult(up, *x1, m - 1);
  }
  *d1 = ts_mult(up, *x1, m - 1);

  return;
}
//****************************************************************************80

__device__ static void legendre_compute_glr(int n, double* __restrict__ x, double* __restrict__ w)

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
//    Output, double X[N], the abscissas.
//
//    Output, double W[N], the weights.
//
{
  if (threadIdx.x == 0) {
    double p;
    double pp;

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
    w[i] = 2.0 / (1.0 - x[i]) / (1.0 + x[i]) / w[i] / w[i];
  }

  __syncthreads();

  double w_sum = 0.0;
  for (int i = 0; i < n; i++) {
    w_sum = w_sum + w[i];
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    w[i] = 2.0 * w[i] / w_sum;
  }
  __syncthreads();
}
}  // namespace gpu
}  // namespace neonufft
