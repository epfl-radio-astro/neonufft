// no include guard for hwy/foreach_target.h to work

#include "neonufft/config.h"

// Disable scalable vector architetures, because we rely on vector lengths at compile time for
// optizations
#ifndef HWY_DISABLED_TARGETS 
#define HWY_DISABLED_TARGETS (HWY_SVE|HWY_SVE2)
#endif

#ifndef HWY_TARGET_INCLUDE
#error "HWY_TARGET_INCLUDE must be defined before including this header file"
#endif

#ifdef NEONUFFT_MULTI_ARCH

// must come before highway.h
#include <hwy/foreach_target.h>
// -----
#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

#undef NEONUFFT_EXPORT_AND_DISPATCH_T
#undef NEONUFFT_DISPATCH
#undef NEONUFFT_EXPORT_FUNC

#define NEONUFFT_EXPORT_AND_DISPATCH_T(arg) HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(arg)
#define NEONUFFT_DISPATCH(arg) HWY_DYNAMIC_DISPATCH(arg)
#define NEONUFFT_EXPORT_FUNC(arg) HWY_EXPORT(arg)

#else

#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

#undef NEONUFFT_EXPORT_AND_DISPATCH_T
#undef NEONUFFT_DISPATCH
#undef NEONUFFT_EXPORT_FUNC

#define NEONUFFT_EXPORT_AND_DISPATCH_T(arg) HWY_STATIC_DISPATCH(arg)
#define NEONUFFT_DISPATCH(arg) HWY_STATIC_DISPATCH(arg)
#define NEONUFFT_EXPORT_FUNC(arg)

#endif

// define a fixed size vector type
namespace neonufft {
namespace {
namespace HWY_NAMESPACE { // required: unique per target
#if HWY_HAVE_SCALABLE
// Use fixed vector length for SVE / SVE2
// Highway guarantees fixed tag size of 16 / sizeof(T)
// Most SVE2 hardware implementations use vector size of 16 bytes (128 bits)
template <typename T>
using TagType = ::hwy::HWY_NAMESPACE::FixedTag<T, 16 / sizeof(T)>;
#else
template <typename T> using TagType = ::hwy::HWY_NAMESPACE::ScalableTag<T>;
#endif
} // namespace HWY_NAMESPACE
} // namespace
} // namespace neonufft
