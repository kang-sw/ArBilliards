#pragma once
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <shared_mutex>

template <typename FnSig>
class thread_pool
{
public: /* Type definitions */
  using self_type = thread_pool<FnSig>;
  using function_type = std::function<FnSig>;
  using task_queue_type = std::queue<function_type>;
  using worker_type = std::pair<std::thread, std::atomic_bool>;
  using worker_container_type = std::vector<function_type>;

public: /* APIs */
  thread_pool(size_t num_initial_workers) noexcept;
  template <typename Fnc>
  void queue_async(Fnc&& func);
  size_t num_workers() const;
  size_t num_free_workers() const;
  void set_num_workers(size_t new_number);
  size_t num_pending_tasks() const;

private:
  task_queue_type tasks_;
  worker_container_type workers_;
  std::atomic_size_t num_running_worker_ = 0;
  std::shared_mutex task_queue_lock_;
  std::shared_mutex worker_container_lock_;
  std::condition_variable task_condition_;
  std::shared_mutex task_condition_lock_;
};

template <typename FnSig>
thread_pool<FnSig>::thread_pool(size_t num_initial_workers) noexcept
{
  set_num_workers(num_initial_workers);
}

template <typename FnSig>
template <typename Fnc>
void thread_pool<FnSig>::queue_async(Fnc&& func)
{
  using namespace std;
  {
    unique_lock<shared_mutex> lock(task_queue_lock_);
    tasks_.push(forward<Fnc>(func));
  }
  task_condition_.notify_one();
}

template <typename FnSig>
void thread_pool<FnSig>::set_num_workers(size_t new_number)
{
  using namespace std;

  struct worker_function_type
  {
    void operator()() const
    {
      while (*is_alive) {
        if (self->task_condition_.wait_for(
              self->task_condition_lock_, 100ms, [this]() {
                return self->num_pending_tasks() > 0;
              })) {
          unique_lock<shared_mutex> lock(self->task_queue_lock_);
        }
      }
    }

    self_type* self;
    atomic_bool* is_alive;
  };

  {
    unique_lock<shared_mutex> lock(worker_container_lock_);
    int64_t delta = int64_t(workers_.size()) - int64_t(new_number);

    if (delta >= 0) {
      while (delta--) {
        // Generate workers ...
      }
    } else {
      // Remove workers ...
    }
  }
}
