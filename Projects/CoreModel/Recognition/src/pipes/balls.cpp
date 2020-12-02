#include "balls.hpp"
#include "../amp-math/helper.hxx"
#include "kangsw/trivial.hxx"
#include "pipepp/options.hpp"

#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include <random>
#include <span>

#undef max
#undef min
#define DO_PERSPECTIVE_SEARCH 1

#include "fmt/format.h"
#include "kangsw/ndarray.hxx"

enum { TILE_SIZE = 1024 };

namespace helper {
namespace {
using namespace concurrency;
using namespace concurrency::graphics;
using namespace concurrency::graphics::direct3d;
using namespace cv;
using namespace std;
using namespace billiards;
using namespace billiards::imgproc;
using namespace kangsw;

auto cvt_const_clrspace(float3 rgb) __GPU { return mathf::rgb2lab(rgb) * float3(1 / 100.f, 1 / 256.f, 1 / 256.f) + float3(0, 0.5f, 0.5f); }

struct kernel_shader {
    // �Է� Ŀ���Դϴ�. X, Y, Z ��ǥ�� ��Ÿ���ϴ�.
    array_view<float_3> const& kernel;

    // Ŀ�� �߽��� ī�޶� ���� Ʈ�������Դϴ�.
    Vec3f const& kernel_center;

    // ������ ī�޶� ���� Ʈ�������Դϴ�.
    Vec3f light_pos;

    // ������ ������Ƽ ���
    Vec3f light_rgb;

    // ī�޶� ����
    float const  cx, cy;
    float const  amplify;
    float3 const proj_matx[3];

    // ���� ������Ƽ ���
    Vec3f const& base_rgb;
    Vec3f const& fresnel0;
    float const  roughness;

    float const ball_radius;

    // ��� Ŀ���Դϴ�.
    // ���̴� �ݵ�� kernel�� ���ƾ� �մϴ�.
    array_view<float_2> const& kernel_pos_buf;
    array_view<float_3> const& kernel_rgb_buf;

    void operator()()
    {
        CV_Assert(kernel_pos_buf.extent.size() == kernel_rgb_buf.extent.size()
                  && kernel_pos_buf.extent.size() == kernel.extent.size());

        // Eye position�� �Ͻ������� �����Դϴ�.
        auto _kcp = value_cast<float3>(kernel_center);
        auto _lp  = value_cast<float3>(light_pos);

        auto _bclor   = value_cast<float3>(base_rgb);
        auto _lclor   = value_cast<float3>(light_rgb);
        auto _fresnel = value_cast<float3>(fresnel0);
        auto _m       = roughness;

        auto& _kernel_src   = kernel;
        auto& _o_kernel_pos = kernel_pos_buf;
        auto& _o_kernel_rgb = kernel_rgb_buf;

        auto   _amp = amplify;
        auto   _cx = cx, _cy = cy;
        float3 _matx0 = proj_matx[0];
        float3 _matx1 = proj_matx[1];
        float3 _matx2 = proj_matx[2];

        // Ŀ���� �߽� ���
        auto _kcent = _matx0 * _kcp + _matx1 * _kcp + _matx2 * _kcp;
        _kcent      = (_kcent / _kcent.z - float3(cx, cy, 0));

        auto _brad = ball_radius;
        parallel_for_each(
          kernel.extent,
          [=](index<1> idx) restrict(amp) {
              namespace m_  = mathf;
              namespace fm_ = fast_math;
              // note. ī�޶� ��ġ�� �׻� �����Դϴ�.
              float3 kp = _kernel_src[idx];
              float3 n  = m_::normalize(kp);
              float3 p  = kp + _kcp;

              // �ü� ������ ���� v�� ��ְ� �ݴ� ������ ���� ���, �������ݴϴ�.
              // �׻� Z=-1 ������ ���� �մϴ�.
              float head_dir_cos;
              if constexpr (DO_PERSPECTIVE_SEARCH) {
                  head_dir_cos = m_::dot(-p, n);
              } else {
                  head_dir_cos = m_::dot(-float3(0, 0, p.z), n);
              }

              if (head_dir_cos < 0.f) { //*/
                  p = _kcp - kp;
                  n = -n;
              }

              float3 v = m_::normalize(-p);
              float3 L = m_::normalize(_lp - p);
              float3 I = -L;
              float3 r = I - 2 * m_::dot(n, I) * n;
              float3 h = m_::normalize(v + L);

              float lambert = m_::Max(m_::dot(L, n), 0.f);

              // �л� ���� ���
              float3 c_d = _lclor * _bclor * lambert;

              // �ݿ� ���� ���
              auto   f0  = 1 - m_::Max(m_::dot(L, h), 0.f);
              float3 R_F = _fresnel + (1 - _fresnel) * f0 * f0 * f0 * f0 * f0;

              // �����Ͻ� ���
              auto   m = (1 - _m) * 256.f;
              float3 S = (m + 8.f) * (1.f / 8.f) * fm_::powf(m_::Max(0.f, m_::dot(n, h)), m);

              // ������ �� �����Ͻ� ����
              float3 c_s = lambert * _lclor * (R_F * S);

              // ���� ���� - ����
              float3 C = c_d + c_s;

              // ��ġ  ����
              if constexpr (DO_PERSPECTIVE_SEARCH) {
                  // ���� ���� ����, ��� amplifier �����մϴ�.
                  float3 sp          = _matx0 * p + _matx1 * p + _matx2 * p;
                  auto   np          = ((sp / sp.z).xy - float2(_cx, _cy) - _kcent.xy);
                  _o_kernel_pos[idx] = np * p.z / _brad * _amp;
              } else {
                  _o_kernel_pos[idx] = (p.xy - _kcp.xy) / _brad; // orthogonal projection
              }

              // ���� ����
              _o_kernel_rgb[idx] += C;
          });
    }
};

void rgb2yuv(array_view<float3> arv)
{
    parallel_for_each(
      arv.extent,
      [=](index<1> idx) restrict(amp) {
          arv[idx] = mathf::rgb2yuv(arv[idx]);
      });
}
} // namespace
} // namespace helper

template <typename Ty_, int Rank_ = 1>
using gpu_array = concurrency::array<Ty_, Rank_>;
using std::optional;
using std::vector;
using namespace concurrency::graphics::direct3d;

struct point_light_t {
    cv::Vec3f pos;
    cv::Vec3f rgb;
};

struct billiards::pipes::ball_finder_executor::impl {
    vector<cv::Vec3f>     pkernel_src;
    vector<cv::Vec2f>     nkernel_src;
    vector<point_light_t> lights;

    optional<gpu_array<float3>> pkernel_src_coords;

    optional<gpu_array<float2>> nkernel_coords;
    optional<gpu_array<float2>> pkernel_coords;

    optional<gpu_array<float3>> pkernel_colors;

    optional<gpu_array<float, 2>> _mbuf_interm_suits;

    vector<cv::Vec2i> _mbuf_pos;
    vector<float>     _mbuf_distance;
    vector<float>     _mbuf_suits;
};

template <typename T, typename R> requires std::is_integral_v<T> auto n_align(T V, R N) { return ((V + N - 1) / N) * N; }

void billiards::pipes::ball_finder_executor::_update_kernel_by(
  pipepp::execution_context& ec, billiards::recognizer_t::frame_desc const& imdesc,
  cv::Vec3f loc_table_pos, cv::Vec3f loc_table_rot, cv ::Vec3f loc_sample_pos)
{
    using namespace std;
    using namespace cv;
    using namespace imgproc;
    auto  ball_color = colors::base_rgb(ec);
    auto  fresnel0   = colors::fresnel0(ec);
    auto  roughness  = colors::roughness(ec);
    auto  ambient    = kangsw::value_cast<float3>(colors::lights::ambient_rgb(ec));
    auto& m          = *_m;

    array<point_light_t, N_MAX_LIGHT> _lights;
    copy(m.lights.begin(), m.lights.end(), _lights.begin());
    span lights{_lights.begin(), m.lights.size()};

    Vec3f pivot_pos = loc_table_pos;
    Vec3f pivot_rot = loc_table_rot;
    auto  tr        = get_transform_matx_fast(pivot_pos, pivot_rot);

    // projection�� ���� �غ� ..
    auto& cam       = imdesc.camera;
    float ball_rad  = kernel::ball_radius(ec);
    auto [cmatx, _] = get_camera_matx(imdesc);
    ;
    float  cx = cam.cx, cy = cam.cy;
    float3 cam_r0 = (float3)kangsw::value_cast<double3>(submatx<0, 0, 1, 3>(cmatx));
    float3 cam_r1 = (float3)kangsw::value_cast<double3>(submatx<1, 0, 1, 3>(cmatx));
    float3 cam_r2 = (float3)kangsw::value_cast<double3>(submatx<2, 0, 1, 3>(cmatx));

    for (auto& l : lights) {
        auto v = tr * concat_vec(l.pos, 1.f);
        l.pos  = subvec<0, 3>(v);
    }

    // Apply ambient light first
    using namespace concurrency;
    using namespace kangsw;
    array_view colors = *_m->pkernel_colors;
    parallel_for_each(
      colors.extent,
      [=, bc_ = kangsw::value_cast<float3>(ball_color)] //
      (index<1> idx) restrict(amp) {
          colors[idx] = bc_ * ambient;
      });

    helper::kernel_shader ks = {
      .kernel        = *m.pkernel_src_coords,
      .kernel_center = loc_sample_pos,

      .cx        = cx,
      .cy        = cy,
      .amplify   = kernel::persp_linear_distance_amp(ec),
      .proj_matx = {cam_r0, cam_r1, cam_r2},

      .base_rgb    = ball_color,
      .fresnel0    = fresnel0,
      .roughness   = roughness,
      .ball_radius = ball_rad,

      .kernel_pos_buf = *m.pkernel_coords,
      .kernel_rgb_buf = *m.pkernel_colors,
    };

    for (auto const& l : lights) {
        ks.light_pos = l.pos;
        ks.light_rgb = l.rgb;

        ks();
    }
}

namespace billiards {
namespace {
float evaluate_suitability(float3 col0, float3 col1, float3 weight, float err_base) __GPU
{
    using namespace concurrency;
    float3 diff    = col1 - col0;
    diff           = diff * diff * weight;
    float distance = fast_math::sqrtf(diff.x + diff.y + diff.z);
    return fast_math::powf(err_base, -distance);
}
} // namespace
} // namespace billiards

void billiards::pipes::ball_finder_executor::_internal_loop(pipepp::execution_context& ec, billiards::pipes::ball_finder_executor::input_type const& in, billiards::pipes::ball_finder_executor::output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    CV_Assert(_m->pkernel_src.size() == _m->nkernel_src.size());

    using std::span;
    using namespace cv;
    using namespace concurrency;
    using namespace imgproc;
    using namespace kangsw::counters;
    using namespace kangsw::misc;

    // �߽� �ȼ� ������ �׸��� �����ο� ����ϴ�.
    Mat1b center_mask;
    auto  image_size = in.domain.size();
    auto  grid_size  = match::optimize::grid_size(ec);
    {
        auto s  = in.center_area_mask.size();
        auto nx = n_align(s.width, grid_size) - s.width;
        auto ny = n_align(s.height, grid_size) - s.height;

        if (nx || ny) {
            cv::copyMakeBorder(in.center_area_mask, center_mask, 0, ny, 0, nx, cv::BORDER_CONSTANT);
        } else {
            center_mask = in.center_area_mask.isContinuous() ? in.center_area_mask : in.center_area_mask.clone();
        }
    }

    // ���� �񱳸� �����ϴ� ������ ... ������ ��ȯ
    cv::Mat3f             domain = in.domain.isContinuous() ? in.domain : in.domain.clone();
    array_view<float3, 2> u_domain(domain.rows, domain.cols, ptr_cast<float3>(&domain(0)));
    parallel_for_each(u_domain.extent,
                      [=](index<2> idx) __GPU {
                          u_domain[idx] = helper::cvt_const_clrspace(u_domain[idx]);
                      });

    if (debug::show_debug_mat(ec)) {
        u_domain.synchronize();
        cv::Mat channels[3];
        split(domain, channels);
        PIPEPP_STORE_DEBUG_DATA("Color converted domain img", (Mat)domain);
        PIPEPP_STORE_DEBUG_DATA("Color converted domain ch 0", channels[0]);
        PIPEPP_STORE_DEBUG_DATA("Color converted domain ch 1", channels[1]);
        PIPEPP_STORE_DEBUG_DATA("Color converted domain ch 2", channels[2]);
    }

    // ���̺� ���
    auto  table_plane = plane_t::from_rp(in.table_rot, in.table_pos, {0, 1, 0});
    auto& imdesc      = *in.p_imdesc;
    plane_to_camera(imdesc, table_plane, table_plane);
    Vec3f local_table_pos = in.table_pos, local_table_rot = in.table_rot;
    world_to_camera(imdesc, local_table_rot, local_table_pos);

    array_view _pkernel_coords = *_m->pkernel_coords;
    array_view _pkernel_colors = *_m->pkernel_colors;
    array_view _nkernel_coords = *_m->nkernel_coords;
    array_view _interm_suits   = *_m->_mbuf_interm_suits;

    // Vector zipping ...
    size_t max_cands_per_grid       = match::optimize::max_pixels_per_grid(ec);
    size_t max_concurrent_iteration = match::optimize::max_concurrent_kernels(ec);
    Size   grid_counts(center_mask.cols / grid_size, center_mask.rows / grid_size);
    auto&  sample_coords      = _m->_mbuf_pos;
    auto&  sample_distance    = _m->_mbuf_distance;
    auto&  sample_suitability = _m->_mbuf_suits;
    sample_coords.reserve(max_cands_per_grid * grid_counts.area()), sample_coords.clear();
    sample_distance.reserve(max_cands_per_grid * grid_counts.area()), sample_distance.clear();
    sample_suitability.reserve(max_cands_per_grid * grid_counts.area()), sample_suitability.clear();

    // Intermediate buffer - ��� ����� ������ �����Դϴ�.
    // grid_size*grid_size�� extent�� �ִ�ġ��, �����δ� �ش� �׸��� ���� ��ȿ �ȼ� ������ŭ�� ���˴ϴ�.
    int n_tot_kernels = _m->pkernel_src.size();
    int n_tot_tiles   = n_tot_kernels / TILE_SIZE;

    // ����� ���� ���� ...
    bool const                      show_grid_rep = debug::show_grid_representation(ec);
    Mat3f                           debug;
    optional<array_view<float3, 2>> u_debug;
    if (show_grid_rep) {
        in.debug_mat.convertTo(debug, CV_32FC3, 1 / 255.f);
        u_debug.emplace(debug.rows, debug.cols, debug.ptr<float3>());
    }

    // TODO ... base color convert(for nkernel)
    auto ball_color  = helper::cvt_const_clrspace(value_cast<float3>(colors::base_rgb(ec)));
    auto ball_radius = kernel::ball_radius(ec);

    auto negative_weight = match::negative_weight(ec);
    auto err_weight      = value_cast<float3>(normalize(match::error_weight(ec)));
    auto err_base        = match::error_base(ec);

    // GRID operation ����
    size_t                        grid_pxl_threshold = std::max<size_t>(1, match::optimize::grid_area_threshold(ec) * grid_size * grid_size);
    std::queue<array_view<float>> pending_suits; // �� ������ ~array_view()���� sync ȣ��Ǵ� ���� ���� ����

    Rect  roi_grid{0, 0, grid_size, grid_size};
    Mat1b grid;

    std::vector<Vec2i> _mcache_points;
    _mcache_points.reserve(roi_grid.area());
    for (auto gidx : counter(grid_counts.height, grid_counts.width)) //
    {
        roi_grid.y = gidx[0] * grid_size;
        roi_grid.x = gidx[1] * grid_size;
        grid       = center_mask(roi_grid);

        // -- �׸��� ������ ��ȿ �ȼ� ī��Ʈ
        auto start_idx = sample_coords.size();

        _mcache_points.clear();
        findNonZero(grid, _mcache_points);
        if (_mcache_points.size() < grid_pxl_threshold) { continue; }

        // ���� �ټ��� �׸��尡 ��ȿ�� ������ �ǴܵǸ�, ����Ʈ�� �ԾƳ��ϴ�.
        while (_mcache_points.size() > max_cands_per_grid) {
            auto idx = rand() % _mcache_points.size();
            std::swap(_mcache_points.back(), _mcache_points[idx]);
            _mcache_points.pop_back();
        }

        for (auto& mp : _mcache_points) {
            sample_coords.emplace_back(mp + (Vec2i)roi_grid.tl());
        }

        auto n_valid_pxls = _mcache_points.size();
        sample_distance.insert(sample_distance.end(), n_valid_pxls, {});
        sample_suitability.insert(sample_suitability.end(), n_valid_pxls, {});

        span pos{sample_coords.begin() + start_idx, n_valid_pxls};
        span dists{sample_distance.begin() + start_idx, n_valid_pxls};
        span suits{sample_suitability.begin() + start_idx, n_valid_pxls};

        //  +- ��ȿ �ȼ��� ���� �ۼ�Ʈ �����̸� �׸��带 discard
        while (pending_suits.size() > max_concurrent_iteration) { pending_suits.front().synchronize(), pending_suits.pop(); }

        // -- ��ȿ �ȼ� ������ �Ÿ� ���(��鿡 ����)
        Vec3f grid_center_pos   = {0, 0, 0}; // ��ǥ ������ ����� �������� ����ϴ�.
        Vec2i screen_center_pos = {0, 0};    // �׸��� ������ �ȼ� ��ǥ ǥ��
        float mean_distance     = 0;
        for (auto [p, d] : kangsw::zip(pos, dists)) {
            Vec3f dest(p[0], p[1], 10.f); // 10���� �̻��� �ȼ��� ������� �ʽ��ϴ�.
            get_point_coord_3d(imdesc, dest[0], dest[1], 10.f);
            if (auto pt = table_plane.find_contact({}, dest)) {
                // ���� ��ǥ�� ��ȯ
                grid_center_pos += *pt;
                screen_center_pos += p;
                mean_distance += d = (*pt)[2]; // �Ÿ� ����
            }
        }
        grid_center_pos /= (float)n_valid_pxls;
        screen_center_pos /= (float)n_valid_pxls;
        mean_distance /= n_valid_pxls;

        // -- �׸����� �������κ���, ���� ��ǥ ����� Ŀ�� ������Ʈ
        _update_kernel_by(ec, imdesc, local_table_pos, local_table_rot, grid_center_pos);
        auto ball_pixel_radius = get_pixel_length(imdesc, ball_radius, 1.0f);

        // �ʿ��ϴٸ�, ����� ������ ������
        if (show_grid_rep) {
            // �׸��� 3D ���� ��� ǥ��
            parallel_for_each(
              _pkernel_coords.extent,
              [=, target = *u_debug,
               local_center   = (float2)kangsw::value_cast<int2>(screen_center_pos),
               pkernel_coords = array_view{_pkernel_coords},
               pkernel_colors = array_view{_pkernel_colors}] //
              (index<1> sample) restrict(amp) {
                  auto kernel_pos   = pkernel_coords[sample];
                  auto kernel_color = pkernel_colors[sample];

                  kernel_pos  = kernel_pos / mean_distance * ball_pixel_radius;
                  auto center = int2(local_center + kernel_pos);

                  if (0 <= center.x && center.x < image_size.width && //
                      0 <= center.y && center.y < image_size.height)  //
                  {
                      target(center.y, center.x) = kernel_color;
                  }
              });

            // Center candidate �� ǥ��
            parallel_for_each(
              extent<1>((int)pos.size()),
              [       =,
               target = *u_debug,
               coords = array_view{(int)pos.size(), ptr_cast<int2>(pos.data())}] //
              (index<1> idx) __GPU                                               //
              {
                  auto coord = coords[idx];
                  target(coord.y, coord.x) += 0.15f;
              });
        }

        // Ŀ�� ������ ��ȯ
        parallel_for_each(_pkernel_colors.extent,
                          [=](index<1> idx) __GPU {
                              _pkernel_colors[idx] = helper::cvt_const_clrspace(_pkernel_colors[idx]);
                          });

        // -- �� ��ȿ �ȼ��� ���� Ŀ�� ��Ī ����
        // ��� ������ �� ����, ~array_view() ȣ�� �� ����ȭ �����ϱ� ���� ť�� ���� �־��
        auto&     o_suits = pending_suits.emplace((int)suits.size(), (float*)suits.data());
        extent<2> interm_extent(n_valid_pxls, n_tot_kernels);
        parallel_for_each(
          interm_extent.tile<1, TILE_SIZE>(),
          [=, pcolors = _pkernel_colors,
           coords              = array_view{(int)pos.size(), ptr_cast<int2>(pos.data())},
           distances           = array_view{(int)dists.size(), dists.data()},
           suitability_divider = float(n_tot_kernels / 2.0f)] //
          (tiled_index<1, TILE_SIZE> tidx) __GPU_ONLY         //
          {
              tile_static float tile_suits[TILE_SIZE];
              auto              loc_tile_idx = tidx.local[1];
              auto              smpl_idx     = tidx.global[0];
              auto              knel_idx     = tidx.global[1];

              auto _coord  = coords[smpl_idx];
              auto pkcoord = _pkernel_coords[knel_idx];
              auto nkcoord = _nkernel_coords[knel_idx];
              auto pcoord  = _coord + int2(pkcoord / distances[smpl_idx] * ball_pixel_radius);
              auto ncoord  = _coord + int2(nkcoord / distances[smpl_idx] * ball_pixel_radius);

              float suit = 0;

              // ��輱�� �������� �����մϴ�.
              if (pcoord.y < 0) { pcoord.y = -pcoord.y; }
              if (pcoord.y >= image_size.height) { pcoord.y = 2 * image_size.height - pcoord.y; }
              if (pcoord.x < 0) { pcoord.x = -pcoord.x; }
              if (pcoord.x >= image_size.width) { pcoord.x = 2 * image_size.width; }

              if (0 <= pcoord.x && pcoord.x <= image_size.width && //
                  0 <= pcoord.y && pcoord.y <= image_size.height)  //
              {
                  auto pcolor = pcolors[knel_idx];
                  auto dcolor = u_domain(pcoord.y, pcoord.x);
                  suit += evaluate_suitability(pcolor, dcolor, err_weight, err_base);
              }

              if (0 <= ncoord.x && ncoord.x <= image_size.width && //
                  0 <= ncoord.y && ncoord.y <= image_size.height)  //
              {
                  auto ncolor = ball_color;
                  auto dcolor = u_domain(ncoord.y, ncoord.x);
                  suit -= negative_weight * evaluate_suitability(ncolor, dcolor, err_weight, err_base);
              }

              tile_suits[loc_tile_idx] = suit;

              tidx.barrier.wait_with_tile_static_memory_fence();
              // tile_suits �޸� ���� --> interm_suits �� ���Կ� ����

              // �� [Ÿ��]�� �Ǻ� Ŀ�� �����忡 ����...
              if (tidx.local[0] == 0 && tidx.local[1] == 0) {
                  float suit_sum = 0;
                  for (int i = 0; i < tidx.tile_dim1; ++i) {
                      suit_sum += tile_suits[i];
                  }

                  _interm_suits(tidx.tile) = suit_sum;
              }

              tidx.barrier.wait_with_all_memory_fence();
              // interm_suits �޸� ���� --> o_suits�� ����

              // �� [����(Dim0)]�� �Ǻ� Ŀ�� �����忡 ���� ...
              if (tidx.global[1] == 0) {
                  float suit_sum = 0;
                  for (int i = 0; i < n_tot_tiles; ++i) {
                      suit_sum += _interm_suits(tidx.global[0], i);
                  }
                  o_suits(tidx.global[0]) = suit_sum / suitability_divider;
              }
          });
    }

    while (pending_suits.empty() == false) { pending_suits.front().synchronize(), pending_suits.pop(); }
    if (show_grid_rep) {
        u_debug->synchronize();
        for (auto gidx : counter(center_mask.rows / grid_size, center_mask.cols / grid_size)) //
        {
            roi_grid.y = gidx[0] * grid_size;
            roi_grid.x = gidx[1] * grid_size;
            rectangle(debug, roi_grid + Size(1, 1), {0.25, 0.25, 0.25});
        }
        PIPEPP_STORE_DEBUG_DATA("Grid Center Rendering Result", (Mat)debug);
    }

    if (sample_suitability.empty()) { return; }

    if (debug::show_debug_mat(ec)) {
        Mat1f suit_map(image_size, 0);

        for (auto idx : counter(sample_coords.size())) {
            Point pt   = sample_coords[idx];
            auto& suit = sample_suitability[idx];

            suit_map(pt) = suit;
        }

        PIPEPP_STORE_DEBUG_DATA("Suitability Field", (Mat)(suit_map * debug::suitability_map_scale(ec)));
    }

    PIPEPP_STORE_DEBUG_DATA("Number of valid pixels", sample_coords.size());
    PIPEPP_STORE_DEBUG_DATA("Number of kernels", n_tot_kernels);
    PIPEPP_STORE_DEBUG_DATA("Number of tiles", n_tot_tiles);
    PIPEPP_STORE_DEBUG_DATA("Default tile size", (size_t)TILE_SIZE);

    size_t search_range = sample_suitability.size();
    auto   conf_thres   = search::confidence_threshold(ec);
    auto   ampl          = search::conf_amp(ec);
    for (int ballidx = 0, max = search::n_balls(ec); ballidx < max; ++ballidx) //
    {
        // ��� suitability�� iterate��, ���� ���� ������Ʈ�� ã���ϴ�.
        auto max_elem  = std::max_element(sample_suitability.begin(), sample_suitability.begin() + search_range);
        auto max_index = max_elem - sample_suitability.begin();

        if (max_elem == sample_suitability.begin() + search_range) {
            break;
        }

        auto conf = *max_elem * ampl;
        if (conf < conf_thres) { break; }

        auto center = sample_coords[max_index];
        auto depth  = sample_distance[max_index];

        Vec3f spos(center[0], center[1], depth);
        auto  _rot = local_table_rot;
        get_point_coord_3d(imdesc, spos[0], spos[1], spos[2]);
        auto local_ball_pos = spos;
        camera_to_world(imdesc, _rot, spos);

        auto& nelem      = o.positions.emplace_back();
        nelem.confidence = conf;
        nelem.position   = spos;

        PIPEPP_STORE_DEBUG_DATA_DYNAMIC_STR(fmt::format("Pos <{}> ", ballidx).c_str(), spos << ", " << conf);

        int pixel_radius = get_pixel_length(imdesc, ball_radius, depth);
        if (ballidx + 1 < max) { // ���� ���� ���� ������ ã�Ƴ� ���� �ĺ����� �����մϴ�.
            auto discard_amp = search::next_ball_erase_amp(ec);

            auto pack  = kangsw::zip(sample_coords, sample_distance, sample_suitability);
            auto pivot = std::partition(
              pack.begin(), pack.begin() + search_range,
              [&, range = pixel_radius * discard_amp](auto tup) {
                  auto pos = std::get<0>(tup);
                  return cv::norm(center, pos, NORM_L2) > range;
              });

            search_range = pivot - pack.begin();
        }

        if (debug::render_3dresult_on_debug_mat(ec)) {
            // debug mat�� ���� ���� ��ġ�� �������մϴ�.
            _update_kernel_by(ec, imdesc, local_table_pos, local_table_rot, local_ball_pos);
            vector<float2> coords = *_m->pkernel_coords;
            vector<float3> colors = *_m->pkernel_colors;

            Mat3b debug3b = in.debug_mat;
            Rect  img_rect({}, debug3b.size());
            for (auto [coord, color] : kangsw::zip(coords, colors)) {
                Point cent = center + (Vec2i)value_cast<Vec2f>(coord * pixel_radius);
                if (img_rect.contains(cent)) {
                    auto color3b  = (Vec3b)value_cast<Vec3f>(color * 255.f);
                    debug3b(cent) = color3b;
                }
            }
        }

        draw_circle(imdesc, in.debug_mat, ball_radius, spos, {0, 255, 0}, 1);
        circle(in.debug_mat, center, 2, {0, 255, 0}, 1);
    }
}

void billiards::pipes::ball_finder_executor::operator()(pipepp::execution_context& ec, input_type const& in, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace std::literals;
    o.positions.clear();

    if (!in.p_imdesc || search::n_balls(ec) == 0) { return; }
    auto&      imdesc          = *in.p_imdesc;
    bool const is_option_dirty = ec.consume_option_dirty_flag();

    if (debug::show_debug_mat(ec)) {
        PIPEPP_STORE_DEBUG_DATA("Center area mask", (cv::Mat)in.center_area_mask.clone());
    }

    // Ŀ�� �ҽ��� �����մϴ�.
    if (is_option_dirty) {
        // �缺 Ŀ���� 3D �� �����Դϴ�.
        auto n_kernel = n_align(kernel::n_dots(ec), TILE_SIZE);
        auto radius   = kernel::ball_radius(ec);
        _m->pkernel_src.clear();
        _m->nkernel_src.clear();

        std::mt19937 rengine(kernel::random_seed(ec));
        for (auto _ : kangsw::counter(n_kernel)) {
            imgproc::random_vector(rengine, _m->pkernel_src.emplace_back(), radius);
        }

        // ���� Ŀ���� 2D ���ͷ� ����
        auto                           neg_range = kernel::negative_weight_range(ec);
        std::uniform_real_distribution distr(neg_range[0], neg_range[1]);
        for (auto _ : kangsw::counter(n_kernel)) {
            imgproc::random_vector(
              rengine,
              _m->nkernel_src.emplace_back(),
              distr(rengine));
        }

        // �׷��Ƚ� ���ۿ� ���ε�
        _m->nkernel_coords.emplace(n_kernel, kangsw::ptr_cast<float2>(&_m->nkernel_src[0]));
        _m->pkernel_src_coords.emplace(n_kernel, kangsw::ptr_cast<float3>(&_m->pkernel_src[0]));
        _m->pkernel_coords.emplace(n_kernel);
        _m->pkernel_colors.emplace(n_kernel);

        // Ŀ�� �޸� �Ҵ�
        auto grid_size         = match::optimize::grid_size(ec);
        auto n_kernel_size_tot = n_kernel;
        auto n_tiles           = n_kernel_size_tot / TILE_SIZE;
        _m->_mbuf_interm_suits.emplace(int(match::optimize::max_pixels_per_grid(ec)), (int)n_tiles);
        CV_Assert(n_kernel_size_tot % TILE_SIZE == 0);
    }

    // ������ �����մϴ�.
    if (is_option_dirty) {
        // ��� ��ġ�� ȸ���� Ŀ���� �� ������ �����ϰ�, Ŀ�� ��ġ�� ī�޶� ���� ��� ��ġ�� ������Ʈ�մϴ�.
        struct light_t {
            cv::Vec3f rgb, pos;
        };

        using lopts      = colors::lights;
        auto    n_lights = colors::lights::n_lightings(ec);
        light_t lights[5];
        {
            using l  = lopts;
            auto tup = std::make_tuple(l::l0{}, l::l1{}, l::l2{}, l::l3{}, l::l4{});
            kangsw::tuple_for_each(
              tup, [&]<typename Ty_ = l::l0>(Ty_ && arg, size_t i) {
                  lights[i].pos = arg.pos(ec);
                  lights[i].rgb = arg.rgb(ec);
              });
        }

        _m->lights.clear();
        for (auto i : kangsw::counter(n_lights)) {
            _m->lights.emplace_back(lights[i].pos, lights[i].rgb);
        }
    }

    // ���ø� ��Ī�� �����մϴ�.
    //
    // 1. ��ü �̹����� �׸���� �����մϴ�.
    // 2. ���ڶ�� �κ��� copyMakeBorder�� ���� (Reflect
    // 3. center mask �󿡼� �� �׸��带 ROIȭ, 0�� �ƴ� �ȼ��� ã���ϴ�.
    // 4. 0�� �ƴ� �ȼ��� �����ϸ�, �ش� �׸��忡 ���� ���̴� ����� Ŀ���� ����մϴ�.
    PIPEPP_ELAPSE_BLOCK("Iterate grids")
    _internal_loop(ec, in, o);

    // ���� Ŀ�� �׸��� ...
    if (debug::show_realtime_kernel(ec)) {
        PIPEPP_ELAPSE_SCOPE("Render sample kernel on center");

        // Ŀ���� �����մϴ�.
        // 1. ���� ��ġ�� �������� �� ��ǥ�� ������ ���� ���� ��ȯ�ϰ�, �ٽ� ī�޶� �������� �����ɴϴ�.
        // 2. ���� ��ǥ�� ī�޶� �������� �����ɴϴ�.
        // 3. Ŀ�� ����
        using namespace std;
        using namespace cv;
        auto& m = *_m;

        vector<float2> positions;
        vector<float3> colors;

        PIPEPP_ELAPSE_BLOCK("Update kernel then download")
        {
            Vec3f loc_pos = in.table_pos, loc_rot = in.table_rot;
            Vec3f kernel_pos = loc_pos;
            imgproc::world_to_camera(imdesc, loc_rot, loc_pos);
            if (o.positions.empty() == false) {
                kernel_pos = o.positions.front().position;
                loc_rot    = in.table_rot;
                imgproc::world_to_camera(imdesc, loc_rot, kernel_pos);
            }
            _update_kernel_by(ec, imdesc, loc_pos, loc_rot, kernel_pos);

            positions = *m.pkernel_coords;
            colors    = *m.pkernel_colors;
        }

        PIPEPP_ELAPSE_SCOPE("Render kernel on sample space");
        auto  scale = debug::kernel_display_scale(ec);
        Mat3f target(scale, scale, {0.4, 0.4, 0.4});
        int   mult = (scale / 4);
        int   ofst = scale / 2;
        int   rad  = 0; // scale / 100;

        Rect bound({}, Size(scale, scale));
        for_each_threads(kangsw::counter(positions.size()), [&](size_t i) {
            auto _p = positions[i];
            auto _c = colors[i];

            auto pos = Vec2i((mult * kangsw::value_cast<Vec2f>(_p)) + Vec2f(ofst, ofst));
            auto col = kangsw::value_cast<Vec3f>(_c);

            if (bound.contains(pos)) {
                target((Point)pos) = col;
            }
        });

        PIPEPP_STORE_DEBUG_DATA("Real time kernel", (Mat)target);
    }
}

void billiards::pipes::ball_finder_executor::link(shared_data& sd, pipepp::execution_context& _exec_cont, input_type& i, pipepp::options& opt)
{
    PIPEPP_REGISTER_CONTEXT(_exec_cont);

    if (sd.table.contour.empty()) {
        i.p_imdesc = nullptr;
        return;
    }

    i.table_pos = sd.table.pos;
    i.table_rot = sd.table.rot;
    sd.rgb.convertTo(i.domain, CV_32FC3, 1 / 255.f);
    i.p_imdesc  = &sd.imdesc_bkup;
    i.debug_mat = sd.debug_mat;

    // �߽� ����� ����ũ�� �����մϴ�.
    using namespace std;
    using namespace cv;
    vector<Vec2i> contour_copy;
    contour_copy.assign(sd.table.contour.begin(), sd.table.contour.end());
    auto roi = boundingRect(contour_copy);

    if (!imgproc::get_safe_ROI_rect(sd.hsv, roi)) {
        i.p_imdesc = nullptr;
        return;
    }

    // ���̺� ���� ������, �� ���� �ش��ϴ� �κи��� ���� �ĺ��� �����մϴ�.
    PIPEPP_ELAPSE_BLOCK("Center Selection")
    {
        Mat1b filtered, area_mask(roi.size(), 0);
        imgproc::range_filter(sd.hsv(roi), filtered,
                              colors::center_area_color_range_lo(opt),
                              colors::center_area_color_range_hi(opt));
        for (auto& pt : contour_copy) { pt -= (Vec2i)roi.tl(); }
        drawContours(area_mask, vector{{contour_copy}}, -1, 255, -1);
        i.center_area_mask.create(i.domain.size());
        copyTo(filtered & area_mask, i.center_area_mask.setTo(0)(roi), {});
    }

    PIPEPP_ELAPSE_SCOPE("Dilate & Erode")
    if (auto n_iter = match::n_center_dilate(opt)) {
        dilate(i.center_area_mask, i.center_area_mask, {}, Point(-1, -1), n_iter);
    }
    if (auto n_iter = match::n_center_erode(opt)) {
        erode(i.center_area_mask, i.center_area_mask, {}, Point(-1, -1), n_iter);
    }

    // ���̺� ����� �ణ ������ϴ�.
    using namespace imgproc;
    float shift = shared_data::ball::offset_from_table_plane(sd);
    auto  up    = rodrigues(sd.table.rot) * Vec3f{0, shift, 0};
    i.table_pos += up;
}

billiards::pipes::ball_finder_executor::ball_finder_executor()
    : _m(std::make_unique<impl>())
{
}

billiards::pipes::ball_finder_executor::~ball_finder_executor() = default;
