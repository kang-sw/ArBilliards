#pragma once

#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

namespace billiards::mathf {
using namespace concurrency;
using namespace concurrency::graphics;
using namespace concurrency::graphics::direct3d;
inline float reduce(float3 v) __GPU { return v.x + v.y + v.z; }

inline float norm(float3 v) __GPU
{
    return fast_math::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

template <typename Ty_>
Ty_ normalize(Ty_ v) __GPU
{
    return v / norm(v);
}

inline float dot(float3 l, float3 r) __GPU
{
    auto s = l * r;
    return s.x + s.y + s.z;
}

template <typename Ty_, typename I_> requires std::is_integral_v<I_>
  Ty_ powi(Ty_ v, I_ val) __GPU
{
    Ty_ r = v;
    while (--val) { r = r * v; }
    return r;
}

inline float3 rgb2yuv(float3 c) __GPU
{
    using f3 = float3;
    return {
      dot(f3(0.299f, 0.587f, 0.114f), c),
      dot(f3(-0.14713f, -0.28886f, 0.436f), c),
      dot(f3(0.615f, -0.51499f, -0.10001f), c)};
}

inline float3 yuv2rgb(float3 c) __GPU
{
    using f3 = float3;
    return {
      dot(f3(1, 0, 1.13983), c),
      dot(f3(1, 0.39465, 0.58060), c),
      dot(f3(1, 2.03211, 0), c)};
}

inline float3 rgb2xyz(float3 c) __GPU
{
    using f3 = float3;
    return f3{
      dot(f3(0.412453f, 0.357580f, 0.180423f), c),
      dot(f3(0.212671f, 0.715160f, 0.072169f), c),
      dot(f3(0.019334f, 0.119193f, 0.950227f), c)};
}

inline float3 rgb2lab(float3 rgb) __GPU
{
    namespace ff = fast_math;
    auto c       = rgb2xyz(rgb) * float3(1 / 0.950456f, 1.f, 1 / 1.088754f);
    auto L       = c.y > 0.008856f ? 116.f * ff::powf(c.y, 1 / 3.f) - 16.f : 903.3f * c.y;

    auto fx = c.x > 0.008856f ? ff::powf(c.x, 1 / 3.f) : 7.787f * c.x + 16.f / 116.f;
    auto fy = c.y > 0.008856f ? ff::powf(c.y, 1 / 3.f) : 7.787f * c.y + 16.f / 116.f;
    auto fz = c.z > 0.008856f ? ff::powf(c.z, 1 / 3.f) : 7.787f * c.z + 16.f / 116.f;

    auto a = 500.f * (fx - fy);
    auto b = 200.f * (fy - fz);

    return {L, a, b};
}

template <typename LTy_>
LTy_ Max(LTy_ const& l, LTy_ const& r) __GPU { return l > r ? l : r; }
template <typename LTy_, typename RTy_>
LTy_ Min(LTy_ const& l, LTy_ const& r) __GPU { return l < r ? l : r; }

inline uint wang_hash(uint seed) __GPU
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

inline namespace types {
struct matx33f {
    auto& operator()(int idx) const __GPU { return val[idx]; }
    auto& operator()(int idx) __GPU { return val[idx]; }

    matx33f() __GPU{};
    matx33f(float3 a, float3 b, float3 c) __GPU { val[0] = a, val[1] = b, val[2] = c; }
    matx33f(const matx33f& r) __GPU { val[0] = r(0), val[1] = r(1), val[2] = r(2); }

    static matx33f eye() __GPU { return {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}; }

    friend matx33f operator*(float a, matx33f const& b) __GPU { return {a * b(0), a * b(1), a * b(2)}; }
    friend matx33f operator*(matx33f const& b, float a) __GPU { return {a * b(0), a * b(1), a * b(2)}; }

    friend float3 operator*(matx33f const& a, float3 const& b) __GPU
    {
        return {
          reduce(a(0) * b),
          reduce(a(1) * b),
          reduce(a(2) * b)
        };
    }

    matx33f operator+(matx33f const& b) const __GPU
    {
        return matx33f((*this)(0) + b(0),
                       (*this)(1) + b(1),
                       (*this)(2) + b(2));
    }

    float3 val[3];
};

} // namespace types

inline matx33f rodrigues(float3 v) __GPU
{
    using namespace concurrency;
    using namespace mathf::types;

    auto O = mathf::norm(v);

    v        = v / O;
    float vx = v.x;
    float vy = v.y;
    float vz = v.z;

    float cosO = fast_math::cos(O);
    float sinO = fast_math::sin(O);

    matx33f V = {{0, -vz, vy}, {vz, 0, -vx}, {-vy, vx, 0}};

    matx33f RRt = {{vx * vx, vx * vy, vx * vz},
                   {vx * vy, vy * vy, vy * vz},
                   {vz * vx, vz * vy, vz * vz}};

    return cosO * matx33f::eye() + sinO * V + (1.f - cosO) * RRt;
}

} // namespace billiards::mathf
