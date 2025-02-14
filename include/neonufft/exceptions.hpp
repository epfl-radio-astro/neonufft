#pragma once

#include <neonufft/config.h>
#include <neonufft/errors.h>

#include <cstddef>
#include <stdexcept>

/*! \cond PRIVATE */
namespace neonufft {
/*! \endcond */

/**
 * A generic error. Base type for all other exceptions.
 */
class NEONUFFT_EXPORT GenericError : public std::exception {
public:
  GenericError() : msg_("Neonufft:: Generic error") {}

  // must be a string literal
  GenericError(const char *msg) : msg_(msg) {}

  const char *what() const noexcept override { return msg_; }

  virtual NeonufftError error_code() const noexcept {
    return NeonufftError::NEONUFFT_GENERIC_ERROR;
  }

private:
  const char *msg_;
};

class NEONUFFT_EXPORT InputError : public GenericError {
public:
  InputError() : GenericError("Neonufft:: Internal error") {}

  // must be a string literal
  InputError(const char *msg) : GenericError(msg) {}

  NeonufftError error_code() const noexcept override {
    return NeonufftError::NEONUFFT_INPUT_ERROR;
  }
};

class NEONUFFT_EXPORT InternalError : public GenericError {
public:
  InternalError() : GenericError("Neonufft:: Internal error") {}

  // must be a string literal
  InternalError(const char *msg) : GenericError(msg) {}

  NeonufftError error_code() const noexcept override {
    return NeonufftError::NEONUFFT_INTERNAL_ERROR;
  }
};

class NEONUFFT_EXPORT MemoryAllocError : public GenericError {
public:
  MemoryAllocError() : GenericError("Neonufft:: Memory allocation error") {}

  // must be a string literal
  MemoryAllocError(const char *msg) : GenericError(msg) {}

  NeonufftError error_code() const noexcept override {
    return NeonufftError::NEONUFFT_MEMORY_ALLOC_ERROR;
  }
};

class NEONUFFT_EXPORT NotImplementedError : public GenericError {
public:
  NotImplementedError() : GenericError("Neonufft:: Not implemented") {}

  // must be a string literal
  NotImplementedError(const char *msg) : GenericError(msg) {}

  NeonufftError error_code() const noexcept override {
    return NeonufftError::NEONUFFT_NOT_IMPLEMENTED_ERROR;
  }
};

/*! \cond PRIVATE */
} // namespace neonufft
/*! \endcond */
