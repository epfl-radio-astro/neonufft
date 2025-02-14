#pragma once

#include <neonufft/config.h>

enum NeonufftError {
  /**
   * Success. No error.
   */
  NEONUFFT_SUCCESS,
  /**
   * Generic error.
   */
  NEONUFFT_GENERIC_ERROR,
  /**
   * Memory error.
   */
  NEONUFFT_MEMORY_ALLOC_ERROR,
  /**
   * Unknown error.
   */
  NEONUFFT_UNKNOWN_ERROR,
  /**
   * Internal error.
   */
  NEONUFFT_INTERNAL_ERROR,
  /**
   * Input handle error.
   */
  NEONUFFT_INPUT_ERROR,
  /**
   * Not Implemented error.
   */
  NEONUFFT_NOT_IMPLEMENTED_ERROR,
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum NeonufftError NeonufftError;
/*! \endcond */
#endif  // cpp
