#pragma once
#include <iterator>
#include <execution>

namespace templates
{
template <typename Ty_>
requires std::is_arithmetic_v<Ty_>&& std::is_integral_v<Ty_> class counter_base
    : public std::iterator<std::random_access_iterator_tag, typename Ty_>
{
public:
    using super = std::iterator<std::random_access_iterator_tag, typename Ty_>;

    counter_base()
        : count_(0) { ; }
    counter_base(Ty_ rhs)
        : count_(rhs) { ; }
    counter_base(counter_base const& rhs)
        : count_(rhs.count_) { ; }

public:
    friend counter_base operator+(counter_base c, typename super::difference_type n) { return counter_base(c.count_ + n); }
    friend counter_base operator+(typename super::difference_type n, counter_base c) { return c + n; }
    friend counter_base operator-(counter_base c, typename super::difference_type n) { return counter_base(c.count_ - n); }
    friend counter_base operator-(typename super::difference_type n, counter_base c) { return c - n; }
    typename super::difference_type operator-(counter_base o) { return count_ - o.count_; }
    counter_base& operator+=(typename super::difference_type n) { return count_ += n, *this; }
    counter_base& operator-=(typename super::difference_type n) { return count_ -= n, *this; }
    counter_base& operator++() { return ++count_, *this; }
    counter_base operator++(int) { return ++count_, counter_base(count_ - 1); }
    counter_base& operator--() { return --count_, *this; }
    counter_base operator--(int) { return --count_, counter_base(count_ - 1); }
    bool operator<(counter_base o) const { return count_ < o.count_; }
    bool operator>(counter_base o) const { return count_ > o.count_; }
    bool operator==(counter_base o) const { return count_ == o.count_; }
    bool operator!=(counter_base o) const { return count_ != o.count_; }
    Ty_ const& operator*() const { return count_; }
    Ty_ const* operator->() const { return &count_; }
    Ty_ const& operator*() { return count_; }
    Ty_ const* operator->() { return &count_; }

private:
    Ty_ count_;
};

template <typename Ty_>
class counter_range_base
{
public:
    counter_range_base(Ty_ min, Ty_ max)
        : min_(min)
        , max_(max) { }

    counter_range_base(Ty_ max)
        : min_(Ty_{})
        , max_(max) { assert(min_ < max_); }

    counter_base<Ty_> begin() const { return min_; }
    counter_base<Ty_> cbegin() const { return min_; }
    counter_base<Ty_> end() const { return max_; }
    counter_base<Ty_> cend() const { return max_; }

private:
    Ty_ min_, max_;
};

using counter = counter_base<size_t>;
using counter_range = counter_range_base<size_t>;

// Executes for_each with given parallel execution policy. However, it returns current partition index within given callback.
// It is recommended to set num_partitions as same as current thread count, but it is not forced.
template <typename It_, typename Fn_, typename ExPo_>
void for_each_partition(ExPo_&&, It_ first, It_ last, Fn_&& cb, size_t num_partitions = std::thread::hardware_concurrency())
{
    assert(first != last);
    assert(num_partitions);
    size_t num_elems = std::distance(first, last);
    size_t steps = (num_elems - 1) / num_partitions + 1;
    num_partitions = std::min(num_elems, num_partitions);
    counter_range partitions(num_partitions);

    std::for_each(
      ExPo_{},
      partitions.begin(),
      partitions.end(),
      [num_elems, steps, &cb, &first](size_t partition_index) {
          It_ it = first, end;
          std::advance(it, steps * partition_index);
          std::advance(end = it, steps * (partition_index + 1) <= num_elems ? steps : num_elems - steps * partition_index);

          for (; it != end; ++it) {
              cb(*it, partition_index);
          }
      });
}
} // namespace templates
