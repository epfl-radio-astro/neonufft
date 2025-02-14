#include "neonufft/config.h"
#include "threading/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>


#ifdef NEONUFFT_TBB
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#elif defined(NEONUFFT_OMP)
#include <omp.h>
#else
#include <thread>
#endif

namespace neonufft {

#ifdef NEONUFFT_TBB
namespace {
class TBBThreadPool : public ThreadPool::ThreadPoolImpl {
public:
  TBBThreadPool(IntType num_threads)
      : num_threads_(num_threads < 1 ? oneapi::tbb::info::default_concurrency()
                                     : num_threads),
        arena_(num_threads_) {
    // TODO: check
    // num_threads_ = arena_.max_concurrency();
  }

  void parallel_for(ThreadPool::ForLoopWrapper &task) override {
    const auto range = task.total_range();

    const auto n_iter = range.end - range.begin;
    const auto iter_block_size = task.iter_block_size();

    arena_.execute([&] {
      oneapi::tbb::parallel_for(
          oneapi::tbb::blocked_range<IntType>(range.begin, range.end,
                                              iter_block_size),
          [&](oneapi::tbb::blocked_range<IntType> r) {
            task.execute(tbb::this_task_arena::current_thread_index(),
                         BlockRange{r.begin(), r.end()});
          });
    });
  }

  IntType num_threads() const override { return num_threads_; }

private:
  IntType num_threads_;
  oneapi::tbb::task_arena arena_;
};

} // namespace

ThreadPool::ThreadPool(IntType num_threads)
    : pool_(std::make_unique<TBBThreadPool>(num_threads)) {}

#elif defined(NEONUFFT_OMP)
namespace {
class OMPThreadPool : public ThreadPool::ThreadPoolImpl {
public:
  OMPThreadPool(IntType num_threads)
      : num_threads_(num_threads){
        if(num_threads_ < 1)
          num_threads_ = omp_get_max_threads();
  }

  void parallel_for(ThreadPool::ForLoopWrapper &task) override {
    const auto range = task.total_range();

    const auto n_iter = range.end - range.begin;
    const auto iter_block_size = task.iter_block_size();

    const auto num_blocks = (n_iter + iter_block_size -1) / iter_block_size;

#pragma omp parallel for num_threads(num_threads_) schedule(dynamic, 1)
    for (IntType idx_block = 0; idx_block < num_blocks; ++idx_block) {
      const IntType thread_id = omp_get_thread_num();

      const auto this_begin = range.begin + idx_block * iter_block_size;
      const auto this_end = std::min(this_begin + iter_block_size, range.end);

      if (this_end > this_begin)
        task.execute(thread_id, BlockRange{this_begin, this_end});
    }
  }

  IntType num_threads() const override { return num_threads_; }

private:
  IntType num_threads_;
};

} // namespace

ThreadPool::ThreadPool(IntType num_threads)
    : pool_(std::make_unique<OMPThreadPool>(num_threads)) {}

#else

namespace {
class NativeThreadPool : public ThreadPool::ThreadPoolImpl {
public:
  NativeThreadPool(IntType num_threads) : num_threads_(num_threads), task_id_(0) {
    if(num_threads_ < 1) {
      num_threads_ = std::thread::hardware_concurrency();
    }

    if (num_threads_ > 1) {
      worker_threads_.reserve(num_threads_ - 1);
    }
    promises_.resize(num_threads_);
    futures_.resize(num_threads_);

    for (IntType id = 1; id < num_threads_; ++id) {
      worker_threads_.emplace_back(
          [this](IntType thread_id) {
            decltype(this->task_id_) last_task_id = 0;
            while (true) {
              std::unique_lock<std::mutex> ul(sync_lock_);
              auto current_task_id = this->task_id_;

              // wait for work or destruction
              cond_var_.wait(ul, [this, &current_task_id, &last_task_id]() {
                current_task_id = this->task_id_;
                return current_task_id != last_task_id;
              });

              // break if no parallel_for available
              if (!(this->for_loop_))
                break;

              last_task_id = current_task_id;

              // allow other threads to continue
              ul.unlock();

              // execute for loop
              this->execute_for_loop(thread_id);
            }
          },
          id);
    }
  };

  IntType num_threads() const override { return num_threads_; }

  void parallel_for(ThreadPool::ForLoopWrapper &task) override {
    // reset block id
    block_id_ = 0;

    // reset promises
    for (IntType id = 0; id < num_threads_; ++id) {
      promises_[id] = std::promise<void>();
      futures_[id] = promises_[id].get_future();
    }

    // signal work ready
    {
      std::lock_guard<std::mutex> guard(sync_lock_);
      for_loop_ = &task;
      ++task_id_;
    }
    cond_var_.notify_all();

    // main thread loop execution
    execute_for_loop(0);

    // Wait for all other threads to finish
    for (IntType id = 1; id < num_threads_; ++id) {
      futures_[id].wait();
    }

    // destroy loop body to avoid dangling reference
    {
      std::lock_guard<std::mutex> guard(sync_lock_);
      for_loop_ = nullptr;
    }

    // check for exceptions. This will throw if any thread threw.
    for (IntType id = 0; id < num_threads_; ++id) {
      futures_[id].get();
    }
  }

  ~NativeThreadPool() {
    // signal end
    {
      std::lock_guard<std::mutex> guard(sync_lock_);
      for_loop_ = nullptr;
      ++task_id_;
    }
    cond_var_.notify_all();
    for (auto &t : worker_threads_) {
      if (t.joinable())
        t.join();
    }
  }

private:
  void execute_for_loop(IntType thread_id) {
    try {
      const auto range = for_loop_->total_range();

      const auto n_iter = range.end - range.begin;

      const auto iter_block_size = this->for_loop_->iter_block_size();

      while (true) {
        const auto this_block_id =
            block_id_.fetch_add(1, std::memory_order_relaxed);

        const auto this_begin = range.begin + this_block_id * iter_block_size;
        const auto this_end = std::min(this_begin + iter_block_size, range.end);

        if (this_end > this_begin)
          this->for_loop_->execute(thread_id, BlockRange{this_begin, this_end});
        else
          break;
      }

      // signal end of task
      this->promises_[thread_id].set_value();
    } catch (...) {
      try {
        // this might throw too. Also signals end of task, but with exception
        // stored
        this->promises_[thread_id].set_exception(std::current_exception());
      } catch (...) {
        std::terminate(); // Can't recover
      }
    }
  }

  IntType num_threads_;
  std::uint64_t task_id_;
  std::atomic<IntType> block_id_;
  std::vector<std::thread> worker_threads_;
  std::vector<std::promise<void>> promises_;
  std::vector<std::future<void>> futures_;
  std::condition_variable cond_var_;
  std::mutex sync_lock_;
  ThreadPool::ForLoopWrapper *for_loop_ = nullptr;
};

} // namespace

ThreadPool::ThreadPool(IntType num_threads)
    : pool_(std::make_unique<NativeThreadPool>(num_threads)) {}

#endif

} // namespace neonufft
