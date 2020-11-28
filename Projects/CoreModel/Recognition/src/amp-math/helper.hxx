#pragma once

#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

namespace billiards::mathf {
using namespace concurrency;
using namespace concurrency::graphics;
using namespace concurrency::graphics::direct3d;

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

float3 rgb2yuv(float3 c) __GPU
{
    using f3 = float3;
    return {
      dot(f3(0.299f, 0.587f, 0.114f), c),
      dot(f3(-0.14713f, -0.28886f, 0.436f), c),
      dot(f3(0.615f, -0.51499f, -0.10001f), c)};
}

float3 yuv2rgb(float3 c) __GPU
{
    using f3 = float3;
    return {
      dot(f3(1, 0, 1.13983), c),
      dot(f3(1, 0.39465, 0.58060), c),
      dot(f3(1, 2.03211, 0), c)};
}

float3 rgb2xyz(float3 c) __GPU
{
    using f3 = float3;
    return {
      dot(c, f3(0.6070f, 0.1740f, 0.2000f)),
      dot(c, f3(0.2990f, 0.5870f, 0.1440f)),
      dot(c, f3(0.0000f, 0.0660f, 1.1120f))};
}

template <typename LTy_>
LTy_ Max(LTy_ const& l, LTy_ const& r) __GPU { return l > r ? l : r; }
template <typename LTy_, typename RTy_>
LTy_ Min(LTy_ const& l, LTy_ const& r) __GPU { return l < r ? l : r; }

} // namespace billiards::mathf
