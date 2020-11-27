#include "balls.hpp"
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

#include "../amp-math/helper.hxx"
#include "kangsw/trivial.hxx"
#include "pipepp/options.hpp"

#undef max
#undef min

enum { NUM_TILES = 256 };

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

struct kernel_shader {
    // �Է� Ŀ���Դϴ�. X, Y, Z ��ǥ�� ��Ÿ���ϴ�.
    array_view<float_3> const& kernel;

    // Ŀ�� �߽��� ī�޶� ���� Ʈ�������Դϴ�.
    Vec3f const& kernel_center;

    // ������ ī�޶� ���� Ʈ�������Դϴ�.
    Vec3f light_pos;

    // ������ ������Ƽ ���
    Vec3f light_rgb;

    // ���� ������Ƽ ���
    Vec3f const& base_rgb;
    Vec3f const& fresnel0;
    float const roughness;

    // ��� Ŀ���Դϴ�.
    // ���̴� �ݵ�� kernel�� ���ƾ� �մϴ�.
    array_view<float_2> const& kernel_pos_buf;
    array_view<float_3> const& kernel_rgb_buf;

    void operator()()
    {
        CV_Assert(kernel_pos_buf.extent.size() == kernel_rgb_buf.extent.size()
                  && kernel_pos_buf.extent.size() == kernel.extent.size());

        // Eye position�� �Ͻ������� �����Դϴ�.
        auto _kcp = value_cast<float3>(-kernel_center);
        auto _lp = value_cast<float3>(light_pos);

        auto _bclor = value_cast<float3>(base_rgb);
        auto _lclor = value_cast<float3>(light_rgb);
        auto _fresnel = value_cast<float3>(fresnel0);
        auto _m = roughness;

        auto& _kernel_src = kernel;
        auto& _o_kernel_pos = kernel_pos_buf;
        auto& _o_kernel_rgb = kernel_rgb_buf;

        parallel_for_each(
          kernel.extent,
          [=](index<1> idx) restrict(amp) {
              namespace m_ = mathf;
              namespace fm_ = fast_math;
              // note. ī�޶� ��ġ�� �׻� �����Դϴ�.
              float3 n = _kernel_src[idx];
              float3 p = n + _kcp;

              // �ü� ������ ���� v�� ��ְ� �ݴ� ������ ���� ���, �������ݴϴ�.
              float3 v = m_::normalize(-p);
              /*if (m_::dot(v, n) < 0.f) {
                  p = p - 2.f * n;
                  n = -n;
                  v = m_::normalize(-p);
              }*/

              float3 L = m_::normalize(_lp - p);

              float3 I = -L;
              float3 r = I - 2 * m_::dot(n, I) * n;

              auto L_dot_n = m_::dot(L, n);
              float lambert = L_dot_n < 0.f ? 0.f : L_dot_n;

              // �л� ���� ���
              float3 c_d = _lclor * _bclor * lambert;

              // �ݿ� ���� ���
              float3 R_F = _fresnel + (1 - _fresnel) * m_::powi(1 - lambert, 5);

              // �����Ͻ� ���
              float3 h = v * L * 0.5f;
              float3 S = (_m + 8.f) * (1.f / 8.f) * fm_::powf(m_::dot(n, h), _m);

              // ������ �� �����Ͻ� ����
              float3 c_s = lambert * _bclor * (R_F * S);

              // ���� ���� - ����
              float3 C = c_d + c_s;

              // Ŀ���� ������ ��ȯ: ���� ���

              // �� ����
              _o_kernel_rgb[idx] += C;
              _o_kernel_pos[idx] = p.xy - _kcp.xy; // orthogonal projection
          });
    }
};
} // namespace
} // namespace helper

template <class... Args_>
using gpu_array = concurrency::array<Args_...>;
using std::optional;
using namespace concurrency::graphics::direct3d;

struct point_light_t {
    cv::Vec3f pos;
    cv::Vec3f rgb;
};

struct billiards::pipes::ball_finder_executor::impl {
    std::vector<cv::Vec3f> pkernel_src_;
    std::vector<cv::Vec2f> nkernel_src_;
    std::vector<point_light_t> lights_;

    optional<gpu_array<float3>> pkernel_src_coords_;

    optional<gpu_array<float2>> nkernel_coords_;
    optional<gpu_array<float2>> pkernel_coords_;

    optional<gpu_array<float3>> pkernel_colors_;
};

template <typename T, typename R> requires std::is_integral_v<T> auto n_align(T V, R N) { return (V + N - 1) / N * N; }

void billiards::pipes::ball_finder_executor::operator()(pipepp::execution_context& ec, input_type const& in, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace std::literals;
    if (!in.p_imdesc) {
        o.positions.clear();
        return;
    }

    auto& imdesc = *in.p_imdesc;
    bool const is_option_dirty = ec.consume_option_dirty_flag();

    if (debug::show_debug_mat(ec)) {
        PIPEPP_STORE_DEBUG_DATA("Center area mask", (cv::Mat)in.center_area_mask.clone());
    }

    // Ŀ�� �ҽ��� �����մϴ�.
    if (is_option_dirty) {
        // �缺 Ŀ���� 3D �� �����Դϴ�.
        auto n_kernel = n_align(kernel::n_dots(ec), NUM_TILES);
        auto radius = kernel::ball_radius(ec);
        _m->pkernel_src_.clear();
        _m->nkernel_src_.clear();

        std::mt19937 rengine(kernel::random_seed(ec));
        for (auto _ : kangsw::counter(n_kernel)) {
            imgproc::random_vector(rengine, _m->pkernel_src_.emplace_back(), radius);
        }

        // ���� Ŀ���� 2D ���ͷ� ����
        auto neg_range = kernel::negative_weight_range(ec);
        std::uniform_real_distribution distr(neg_range[0], neg_range[1]);
        for (auto _ : kangsw::counter(n_kernel)) {
            imgproc::random_vector(
              rengine,
              _m->nkernel_src_.emplace_back(),
              distr(rengine) * radius);
        }

        // �׷��Ƚ� ���ۿ� ���ε�
        _m->nkernel_coords_.emplace(n_kernel, kangsw::ptr_cast<float2>(&_m->nkernel_src_[0]));
        _m->pkernel_src_coords_.emplace(n_kernel, kangsw::ptr_cast<float3>(&_m->pkernel_src_[0]));
        _m->pkernel_coords_.emplace(n_kernel);
        _m->pkernel_colors_.emplace(n_kernel);
    }

    // ������ �����մϴ�.
    if (is_option_dirty) {
        // ��� ��ġ�� ȸ���� Ŀ���� �� ������ �����ϰ�, Ŀ�� ��ġ�� ī�޶� ���� ��� ��ġ�� ������Ʈ�մϴ�.
        struct light_t {
            cv::Vec3f rgb, pos;
        };

        using lopts = colors::lights;
        auto n_lights = colors::lights::n_lightings(ec);
        light_t lights[5];
        {
            using l = lopts;
            auto tup = std::make_tuple(l::l0{}, l::l1{}, l::l2{}, l::l3{}, l::l4{});
            kangsw::tuple_for_each(
              tup, [&]<typename Ty_ = l::l0>(Ty_ && arg, size_t i) {
                  lights[i].pos = arg.pos(ec);
                  lights[i].rgb = arg.rgb(ec);
              });
        }

        _m->lights_.clear();
        for (auto i : kangsw::counter(n_lights)) {
            _m->lights_.emplace_back(lights[i].pos, lights[i].rgb);
        }
    }

    // ���ø� ��Ī�� �����մϴ�.
    //
    // 1. ��ü �̹����� �׸���� �����մϴ�.
    // 2. ���ڶ�� �κ��� copyMakeBorder�� ���� (Reflect
    // 3. center mask �󿡼� �� �׸��带 ROIȭ, 0�� �ƴ� �ȼ��� ã���ϴ�.
    // 4. 0�� �ƴ� �ȼ��� �����ϸ�, �ش� �׸��忡 ���� ���̴� ����� Ŀ���� ����մϴ�.
    // 5.
    //

    {
        // Ŀ�� �ʱ�ȭ
        using namespace concurrency;
        array_view colors = *_m->pkernel_colors_;
        parallel_for_each(
          colors.extent,
          [=](index<1> idx) restrict(amp) { colors[idx] = {}; });
    }

    {
        cv::Vec3f world_pos = in.table_pos;
        cv::Vec3f world_rot = in.table_rot;

        auto inv_cam = imdesc.camera_transform.inv();
        auto ball_color = colors::base_rgb(ec);
        auto fresnel0 = colors::fresnel0(ec);
        auto roughness = colors::roughness(ec);
        // Ŀ���� �����մϴ�.
        // 1. ���� ��ġ�� �������� �� ��ǥ�� ������ ���� ���� ��ȯ�ϰ�, �ٽ� ī�޶� �������� �����ɴϴ�.
        // 2. ���� ��ǥ�� ī�޶� �������� �����ɴϴ�.
        // 3. Ŀ�� ����
        using namespace std;
        using namespace cv;
        using namespace imgproc;

        auto& m = *_m;

        array<point_light_t, N_MAX_LIGHT> _lights;
        copy(m.lights_.begin(), m.lights_.end(), _lights.begin());
        span lights{_lights.begin(), m.lights_.size()};

        auto world_tr = get_transform_matx_fast(world_pos, world_rot);
        for (auto& l : lights) {
            l.pos = subvec<0, 3>(inv_cam * (world_tr * concat_vec(l.pos, 1.f)).mul({1, -1.f, 1}));
        }

        Vec3f pivot_pos = world_pos;
        Vec3f pivot_rot = world_rot;
        world_to_camera(imdesc, pivot_rot, pivot_pos);

        helper::kernel_shader ks = {
          .kernel = *m.pkernel_src_coords_,
          .kernel_center = pivot_pos,
          .base_rgb = ball_color,
          .fresnel0 = fresnel0,
          .roughness = roughness,

          .kernel_pos_buf = *m.pkernel_coords_,
          .kernel_rgb_buf = *m.pkernel_colors_,
        };

        for (auto const& l : lights) {
            ks.light_pos = l.pos;
            ks.light_rgb = l.rgb;

            ks();
        }

        // ~done

        if (debug::show_debug_mat(ec)) {
            // Download kernel
            vector<float2> positions = *m.pkernel_coords_;
            vector<float3> colors = *m.pkernel_colors_;

            auto scale = debug::kernel_display_scale(ec);
            Mat3b target(scale, scale, {0, 0, 0});
            int mult = (scale / 4) / kernel::ball_radius(ec);
            int ofst = scale / 2;
            int rad = 0; // scale / 100;

            for (auto [_p, _c] : kangsw::zip(positions, colors)) {
                auto pos = Vec2i((mult * kangsw::value_cast<Vec2f>(_p)) + Vec2f(ofst, ofst));
                auto col = Vec3b(255 * kangsw::value_cast<Vec3f>(_c));

                circle(target, pos, rad, col, -1);
            }

            PIPEPP_STORE_DEBUG_DATA("Real time kernel", (Mat)target);
        }
    }
}

void billiards::pipes::ball_finder_executor::link(shared_data& sd, input_type& i, pipepp::options& opt)
{
    if (sd.table.contour.empty()) {
        i.p_imdesc = nullptr;
        return;
    }

    i.table_pos = sd.table.pos;
    i.table_rot = sd.table.rot;
    i.domain = sd.retrieve_image_in_colorspace(match::color_space(opt));
    i.p_imdesc = &sd.imdesc_bkup;

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
    Mat1b filtered, area_mask(roi.size(), 0);
    imgproc::range_filter(sd.hsv(roi), filtered,
                          colors::center_area_color_range_lo(opt),
                          colors::center_area_color_range_hi(opt));
    for (auto& pt : contour_copy) { pt -= (Vec2i)roi.tl(); }
    drawContours(area_mask, vector{{contour_copy}}, -1, 255, -1);
    i.center_area_mask.create(i.domain.size());
    copyTo(filtered & area_mask, i.center_area_mask.setTo(0)(roi), {});
}

billiards::pipes::ball_finder_executor::ball_finder_executor()
    : _m(std::make_unique<impl>())
{
}

billiards::pipes::ball_finder_executor::~ball_finder_executor() = default;
