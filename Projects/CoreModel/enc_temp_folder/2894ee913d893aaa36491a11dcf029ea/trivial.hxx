#pragma once
#include <span>
#include <vector>
#include <algorithm>
#include <numeric>

namespace kangsw::inline misc {
/**
 * @see https://stackoverflow.com/questions/55288555/c-check-if-statement-can-be-evaluated-constexpr
 * evaluates given expression can be constexpr
 */
template <class Lambda, int = (Lambda{}(), 0)>
constexpr bool is_constexpr(Lambda) { return true; }
constexpr bool is_constexpr(...) { return false; }

/**
 * parameter pack의 N번째 argument를 얻습니다.
 */
template <size_t N, typename... Args>
decltype(auto) get_pack_element(Args&&... as) noexcept {
    return std::get<N>(std::forward_as_tuple(std::forward<Args>(as)...));
}

/**
 * 64-bit compile-time hash
 * @see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
 */
constexpr uint64_t _fnv1a_impl(char const* s, char const* end) {
    constexpr uint64_t PRIME = 0x100000001b3;
    constexpr uint64_t OFFSET = 0xcbf29ce484222325;
    uint64_t hash = OFFSET; // magic number

    for (; s != end; ++s) { hash = (hash ^ *s) * PRIME; }
    return hash;
}
constexpr uint64_t fnv1a(char const* str) {
    char const* h = str;
    for (; *h; ++h) {}
    return _fnv1a_impl(str, h);
}

/**
 * Manages struct ownership
 */
class ownership {
public:
    ownership(bool owning = true) :
        owning_(owning) {}

    ownership(ownership const&) = delete;
    ownership(ownership&& r) noexcept { *this = std::move(r); };
    ownership& operator=(ownership const&) = delete;
    ownership& operator=(ownership&& r) noexcept { return owning_ = r.owning_, r.owning_ = false, *this; };

    operator bool() { return owning_; }

private:
    bool owning_ = false;
};

/**
 * Same sized pointer cast
 */
template <typename DTy_, typename STy_>
auto value_cast(STy_ const& ref) {
    static_assert(sizeof(STy_) == sizeof(DTy_));
    return *reinterpret_cast<DTy_ const*>(&ref);
}

template <typename DTy_, typename STy_>
auto ptr_cast(STy_ const* ref) {
    static_assert(sizeof(STy_) == sizeof(DTy_));
    return reinterpret_cast<DTy_ const*>(ref);
}

template <typename DTy_, typename STy_>
auto ptr_cast(STy_* ref) {
    static_assert(sizeof(STy_) == sizeof(DTy_));
    return reinterpret_cast<DTy_*>(ref);
}

/**
 * Get index of element from reference
 */
template <typename Container_>
auto get_index_of(Container_ const& ct, typename Container_::value_type const& r) {
    auto index = &r - ct.data();
    if (&r < ct.data() || index < 0) { throw; }
    return index;
}

/**
 * count of raw array
 */
template <typename Ty_, size_t N>
constexpr size_t countof(Ty_ (&)[N]) { return N; }

/**
 * swap remove an vector-like container
 */
template <typename Container_>
void swap_remove(Container_& ct, typename Container_::size_type at) {
    if (at < ct.size()) { std::swap(ct[at], ct.back()); }
    ct.pop_back();
}

namespace ___internals {
thread_local static std::vector<size_t> indexes;
}
/**
 * sort vectors by one vector
 */
template <typename Pvt_, typename... Containers_>
auto& sort_index(Pvt_& pivot) {
    auto& indexes = ___internals::indexes;
    std::iota(indexes.begin(), indexes.end(), std::size(pivot));
    std::sort(indexes.begin(), indexes.end(),
              [&pivot, pred = std::less<Pvt_>{}](size_t l, size_t r) { return pred(pivot[l], pivot[r]); });

    return indexes;
}

} // namespace kangsw::inline trivial