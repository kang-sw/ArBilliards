#pragma once
#include <cstdint>
#include <iterator>

namespace billiards
{
template <typename Ty_>
requires std::is_arithmetic_v<Ty_>&& std::is_integral_v<Ty_> class counter
    : public std::iterator<std::random_access_iterator_tag, typename Ty_>
{
public:
    using super = std::iterator<std::random_access_iterator_tag, typename Ty_>;

    counter()
        : count_(0) { ; }
    counter(Ty_ rhs)
        : count_(rhs) { ; }
    counter(counter const& rhs)
        : count_(rhs.count_) { ; }

public:
    friend counter operator+(counter c, typename super::difference_type n) { return counter(c.count_ + n); }
    friend counter operator+(typename super::difference_type n, counter c) { return c + n; }
    friend counter operator-(counter c, typename super::difference_type n) { return counter(c.count_ - n); }
    friend counter operator-(typename super::difference_type n, counter c) { return c - n; }
    typename super::difference_type operator-(counter o) { return count_ - o.count_; }
    counter& operator+=(typename super::difference_type n) { return ++count_, *this; }
    counter& operator-=(typename super::difference_type n) { return --count_, *this; }
    counter& operator++() { return ++count_, *this; }
    counter operator++(int) { return ++count_, counter(count_ - 1); }
    counter& operator--() { return --count_, *this; }
    counter operator--(int) { return --count_, counter(count_ - 1); }
    bool operator<(counter o) const { return count_ < o.count_; }
    bool operator>(counter o) const { return count_ > o.count_; }
    bool operator==(counter o) const { return count_ == o.count_; }
    bool operator!=(counter o) const { return count_ != o.count_; }
    Ty_ const& operator*() const { return count_; }
    Ty_ const* operator->() const { return &count_; }
    Ty_ const& operator*() { return count_; }
    Ty_ const* operator->() { return &count_; }

protected:
    Ty_ count_;
};
} // namespace billiards
