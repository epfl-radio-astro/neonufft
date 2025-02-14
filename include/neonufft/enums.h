#pragma once

#include <neonufft/config.h>


enum NeonufftModeOrder {
  NEONUFFT_MODE_ORDER_FFT,
  NEONUFFT_MODE_ORDER_CMCL,
};

enum NeonufftKernelType { NEONUFFT_ES_KERNEL };

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum NeonufftModeOrder NeonufftModeOrder;
/*! \endcond */
#endif  // cpp
