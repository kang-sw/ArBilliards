#include "balls.hpp"
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

#include "../amp-math/helper.hxx"
#include "pipepp/options.hpp"

#undef max, min

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
    // 입력 커널입니다. X, Y, Z 좌표를 나타냅니다.
    array_view<float_3> const& kernel;

    // 커널 중심의 카메라 공간 트랜스폼입니다.
    Vec3f const& kernel_center;

    // 점광의 카메라 공간 트랜스폼입니다.
    Vec3f const& light_pos;

    // 점광의 프로퍼티 목록
    Vec3f const& light_rgb;

    // 공의 프로퍼티 목록
    Vec3f const& base_rgb;
    Vec3f const& fresnel0;
    float const roughness;

    // 출력 커널입니다.
    // 길이는 반드시 kernel과 같아야 합니다.
    array_view<float_2> const& kernel_pos_buf;
    array_view<float_3> const& kernel_rgb_buf;

    void operator()()
    {
        CV_Assert(kernel_pos_buf.extent.size() == kernel_rgb_buf.extent.size()
                  && kernel_pos_buf.extent.size() == kernel.extent.size());

        // Eye position은 암시적으로 원점입니다.
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
              // note. 카메라 위치는 항상 원점입니다.
              float3 p = _kernel_src[idx];
              float3 n = p - _kcp; // 좌표에서 커널 중심점을 뺀 값 = 노멀, 커널 길이는 항상 0이므로 ...

              // 시선 역방향 벡터 v가 노멀과 반대 방향을 보는 경우, 뒤집어줍니다.
              float3 v = m_::normalize(-p);
              if (m_::dot(v, n) < 0.f) {
                  p = p - 2.f * n;
                  n = -n;
              }

              float3 L = m_::normalize(_lp - p);

              float3 I = -L;
              float3 r = I - 2 * m_::dot(n, I) * n;

              auto L_dot_n = m_::dot(L, n);
              float lambert = L_dot_n < 0.f ? 0.f : L_dot_n;

              // 분산 조명 계산
              float3 c_d = _lclor * _bclor * lambert;

              // 반영 조명 계산
              float3 R_F = _fresnel + (1 - _fresnel) * m_::powi(1 - lambert, 5);

              // 러프니스 계산
              float3 h = v * L * 0.5f;
              float3 S = (_m + 8.f) * (1.f / 8.f) * fm_::powf(m_::dot(n, h), _m);

              // 프레넬 및 러프니스 결합
              float3 c_s = lambert * _bclor * (R_F * S);

              // 광원 조명 - 최종
              float3 C = c_d + c_s;

              // 커널의 색공간 전환: 고정 사용

              // 값 저장
              _o_kernel_rgb[idx] = C;
              _o_kernel_pos[idx] = p.xy; // orthogonal projection
          });
    }
};
} // namespace
} // namespace helper

template <class... Args_>
using gpu_array = concurrency::array<Args_...>;
using std::optional;
using namespace concurrency::graphics::direct3d;

struct billiards::pipes::ball_finder_executor::impl {
    std::vector<cv::Vec3f> pkernel_src_;
    std::vector<cv::Vec2f> nkernel_src_;

    optional<gpu_array<float2>> nkernel_coords_;
    optional<gpu_array<float2>> pkernel_coords_;
    optional<gpu_array<float3>> pkernel_rgb_;
};

template <typename T> requires std::is_integral_v<T>
  T n_align(T V, size_t N) { return (V + N - 1) / N * N; }

void billiards::pipes::ball_finder_executor::operator()(pipepp::execution_context& ec, input_type const& in, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace std::literals;
    if (!in.p_imdesc) {
        o.positions.clear();
        return;
    }

    bool const is_option_dirty = ec.consume_option_dirty_flag();

    if (debug::show_debug_mat(ec)) {
        PIPEPP_STORE_DEBUG_DATA("Center area mask", (cv::Mat)in.center_area_mask.clone());
    }

    // 커널 소스를 갱신합니다.
    if (is_option_dirty) {
        // 양성 커널은 3D 점 집합입니다.
        auto n_kernel = n_align(kernel::n_dots(ec), NUM_TILES);
        _m->pkernel_src_.clear();
        _m->nkernel_src_.clear();

        std::mt19937 rengine(kernel::random_seed(ec));
        for (auto _ : kangsw::counter(n_kernel)) {
            imgproc::random_vector(rengine, _m->pkernel_src_.emplace_back(), 1.f);
        }

        // 음성 커널은 2D 벡터로 정의
        auto neg_range = kernel::negative_weight_range(ec);
        std::uniform_real_distribution distr(neg_range[0], neg_range[1]);
        for (auto _ : kangsw::counter(n_kernel)) {
            imgproc::random_vector(
              rengine,
              _m->nkernel_src_.emplace_back(),
              distr(rengine));
        }

        // 그래픽스 버퍼에 업로드
    }

    // 템플릿 매칭을 수행합니다.
    //
    // 1. 전체 이미지를 그리드로 분할합니다.
    // 2. 모자라는 부분은 copyMakeBorder로 삽입 (Reflect
    // 3. center mask 상에서 각 그리드를 ROI화, 0이 아닌 픽셀을 찾습니다.
    // 4. 0이 아닌 픽셀이 존재하면, 해당 그리드에 대해 셰이더 적용된 커널을 계산합니다.
    // 5.
    //
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

    // 중심 에어리어 마스크를 생성합니다.
    using namespace std;
    using namespace cv;
    vector<Vec2i> contour_copy;
    contour_copy.assign(sd.table.contour.begin(), sd.table.contour.end());
    auto roi = boundingRect(contour_copy);

    if (!imgproc::get_safe_ROI_rect(sd.hsv, roi)) {
        i.p_imdesc = nullptr;
        return;
    }

    // 테이블 영역 내에서, 공 색상에 해당하는 부분만을 중점 후보로 선택합니다.
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

void billiards::pipes::ball_finder_executor::_update_kernel(cv::Vec3f world_pos, cv::Vec3f world_rot)
{
}
