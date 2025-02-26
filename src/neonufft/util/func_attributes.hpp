#pragma once

#include "neonufft/config.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NEONUFFT_H_FUNC __host__
#define NEONUFFT_H_D_FUNC __host__ __device__
#define NEONUFFT_D_FUNC __device__
#else
#define NEONUFFT_H_FUNC
#define NEONUFFT_H_D_FUNC
#endif
