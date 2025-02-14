#pragma once

#include "neonufft/config.h"

#include "neonufft/types.hpp"
#include "neonufft/exceptions.hpp"

#include <memory>
#include <type_traits>

namespace neonufft {

struct BlockRange {
  IntType begin = 0, end = 0;
};

class ThreadPool {
public:
  ThreadPool(IntType num_threads);

  IntType num_threads() const { return pool_->num_threads(); }

  template <typename F>

  // parallel for, where each thread executes at most one BlockRange
  void parallel_for(BlockRange range, F &&func) {
    if (range.begin >= range.end)
      return;

    const auto n_iter = range.end - range.begin;

    const IntType iter_block_size =
        (n_iter + pool_->num_threads() - 1) / pool_->num_threads();
    parallel_for(range, iter_block_size, std::forward<F>(func));
  }

  // parallel for with given block size
  template <typename F>
  void parallel_for(BlockRange range, IntType iter_block_size, F &&func) {
    static_assert(std::is_invocable_v<F, IntType, BlockRange>);

    if (range.begin >= range.end)
      return;

    if(iter_block_size < 1) {
      throw InternalError("parallel_for: iter_block_size < 1");
    }

    // execute on main thread only if number of threads <= 1 or the
    // iter_block_size covers the entire range
    if (pool_->num_threads() <= 1 ||
        iter_block_size >= (range.end - range.begin)) {
      func(0, range);
      return;
    }

    const auto n_iter = range.end - range.begin;

    struct ParallelForImpl : public ForLoopWrapper {
      ParallelForImpl(BlockRange p_range, IntType p_iter_block_size, F &p_func)
          : ForLoopWrapper(p_range, p_iter_block_size), func_(p_func) {}

      void execute(IntType thread_id, const BlockRange &r) override {
        func_(thread_id, r);
      }

      F &func_;
    };

    ParallelForImpl impl(range, iter_block_size, func);

    // execute loop in parallel
    pool_->parallel_for(impl);
  }

  struct ForLoopWrapper {
    explicit ForLoopWrapper(BlockRange r, IntType iter_block_size)
        : total_range_(std::move(r)), iter_block_size_(iter_block_size) {}

    virtual void execute(IntType thread_id, const BlockRange &r) = 0;

    const BlockRange &total_range() const { return total_range_; }

    const IntType &iter_block_size() const { return iter_block_size_; }

    virtual ~ForLoopWrapper() = default;

    BlockRange total_range_;
    IntType iter_block_size_;
  };

  struct ThreadPoolImpl {
    virtual void parallel_for(ForLoopWrapper &) = 0;

    virtual IntType num_threads() const = 0;

    virtual ~ThreadPoolImpl() = default;
  };

private:
  std::unique_ptr<ThreadPoolImpl> pool_;
};
} // namespace neonufft
