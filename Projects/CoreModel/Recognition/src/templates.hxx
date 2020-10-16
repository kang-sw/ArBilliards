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

private:
    using diff_t = typename super::difference_type;

    friend void t()
    {
        return;
    }

public:
    friend counter operator+(counter c, diff_t n) { return counter(c.count_ + n); }
    friend counter operator+(diff_t n, counter c) { return c + n; }
    friend counter operator-(counter c, diff_t n) { return counter(c.count_ - n); }
    friend counter operator-(diff_t n, counter c) { return c - n; }
    diff_t operator-(counter o) { return diff_t(count_ - o.count_); }
    counter& operator+=(diff_t n) { return ++count_, *this; }
    counter& operator-=(diff_t n) { return --count_, *this; }
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
