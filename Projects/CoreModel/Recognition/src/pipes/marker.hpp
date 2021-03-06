#pragma once
#include "../image_processing.hpp"
#include "recognizer.hpp"

namespace pipepp {
class execution_context;
}

namespace billiards::pipes {
using namespace std::literals;

namespace helpers {
struct table_edge_extender {
    // inputs
    cv::Vec3f                  table_rot, table_pos;
    imgproc::img_t const*      p_imdesc;
    std::span<const cv::Vec2f> table_contour;

    // opts
    int    num_insert_contour_vertexes = 5;
    double table_border_range_outer    = 0.06;
    double table_border_range_inner    = 0.06;
    bool   should_draw                 = true;

    cv::Mat const* debug_mat = {};

    struct {
        std::vector<cv::Vec2f> outer_contour;
        std::vector<cv::Vec2f> inner_contour;
    } output;

    void operator()(pipepp::execution_context& ec, cv::Mat& marker_area_mask);
};

} // namespace helpers

/**
 * TODO
 * 마커를 탐색합니다. 모든 흰 점을 대해, Sparse Kernel을 적용해 찾아냅니다.
 * 이 때, 커널의 기본형은 원형의 점 목록을 3D 공간으로 변환하고, 각 점에 버텍스 셰이더를 적용해 얻습니다.
 *
 * @details
 *
 * 희소 커널 원형의 각 버텍스를 X, Z 평면(테이블과 같은 평면)상에 스폰합니다. 테이블의 카메라에 대한 상대 로테이션으로 각 버텍스를 회전시키고 화면에 원근 투영하면, 평면 커널을 획득할 수 있습니다.
 */
PIPEPP_EXECUTOR(table_marker_finder)
{
    PIPEPP_OPTION(show_debug_mats, true);

    PIPEPP_CATEGORY(debug, "Debug")
    {
        PIPEPP_OPTION(kernel_view_size, 200u);
        PIPEPP_OPTION(current_kernel_view_scale, 0.05f);
        PIPEPP_OPTION(depth_view_multiply, 10.f);
        PIPEPP_OPTION(suitability_view_multiply, 10.f);
    };

    PIPEPP_CATEGORY(kernel, "Kernel")
    {
        PIPEPP_OPTION(positive_area,
                      cv::Vec2d(0, 1),
                      u8"중심점으로부터, 양의 가중치로 평가되는 구간입니다.",
                      pipepp::verify::clamp_all<cv::Vec2d>(0, 1) | pipepp::verify::ascending<cv::Vec2d>());
        PIPEPP_OPTION(negative_area,
                      cv::Vec2d(1, 2),
                      u8"중심점으로부터, 음의 가중치로 평가되는 구간입니다.",
                      pipepp::verify::minimum_all<cv::Vec2d>(0) | pipepp::verify::ascending<cv::Vec2d>());
        PIPEPP_OPTION(generator_positive_radius, 10u, "", pipepp::verify::maximum(10000u));
        PIPEPP_OPTION(generator_negative_radius, 10u, "", pipepp::verify::maximum(10000u));
        PIPEPP_OPTION(random_seed, 42u);
    };

    PIPEPP_OPTION(marker_radius, 0.005, "Units in centimeters");

    PIPEPP_CATEGORY(pp, "Preprocessing")
    {
        PIPEPP_OPTION(use_all_non_blue_area, true,
                      u8"True를 지정하면, 현재 당구대의 컨투어 영역이 아닌 파란색 픽셀이 존재하는 모든 영역에서"
                      " 마커 후보를 탐색합니다.");

        PIPEPP_OPTION(num_inserted_contours, 5,
                      u8"마커 탐색을 위해 당구대 각 정점을 확장할 때, "
                      "새롭게 삽입할 컨투어 정점의 개수입니다. 영상 처리 자체에 미치는 영향은 미미합니다.");
        PIPEPP_OPTION(marker_range_outer, 0.1,
                      u8"일반적으로, 당구대의 펠트 경계선부터 바깥쪽까지의 영역 길이를 지정합니다.\n"
                      "meter 단위");
        PIPEPP_OPTION(marker_range_inner, 0.0,
                      u8"당구대의 펠트 경계부터 안쪽으로 마커 영역 마스크를 설정하는 데 사용합니다.\n"
                      "일반적으로 0을 지정하면 충분합니다.\n"
                      "meter 단위");

        PIPEPP_CATEGORY(m2, "All Area")
        {
            PIPEPP_OPTION(approx_epsilon, 1.0,
                          u8"모든 영역의 컨투어를 계산하고 근사할 때적용할 epsilon 계수입니다.");
        };
    };

    PIPEPP_CATEGORY(filter, "Filtering")
    {
        PIPEPP_OPTION(method, 0,
                      "[0] Simple color range filter \n"
                      "[1] Lightness edge detector\n",
                      pipepp::verify::contains(0, 1));

        PIPEPP_OPTION(filter_color_space, "HSV"s, u8"유효 픽셀을 검출할 색공간입니다.", verify::color_space_string_verify);
        PIPEPP_OPTION(convolution_color_space, "HSV"s, u8"마커의 컨볼루션을 적용할 색공간입니다.", verify::color_space_string_verify);

        PIPEPP_CATEGORY(convolution, "Convolution")
        {
            PIPEPP_OPTION(pivot_color, cv::Vec3b(233, 233, 233), u8"마커의 대표 색상입니다. 색 공간에 의존적입니다.");
            PIPEPP_OPTION(pivot_color_weight, cv::Vec3f(1, 1, 1), u8"대표 색상의 적용 질량입니다.");
            PIPEPP_OPTION(color_distance_error_base, 1.01f,
                          u8"거리 계산 함수 d(x) = b^(-x)에 대해, b의 값입니다.",
                          pipepp::verify::minimum(1.f));
            PIPEPP_OPTION(negative_kernel_weight, 1.0f, u8"Negative kernel이 값의 평가에 미치는 영향 정도입니다.",
                          pipepp::verify::minimum(0.f));
        };

        PIPEPP_CATEGORY(method0, "Method 0: Range Filter")
        {
            PIPEPP_OPTION(range_lo, cv::Vec3b(125, 125, 125));
            PIPEPP_OPTION(range_hi, cv::Vec3b(255, 255, 255));
        };

        PIPEPP_CATEGORY(method1, "Method 1: Laplacian Filter")
        {
            PIPEPP_OPTION(enable_gpu, false);
            PIPEPP_OPTION(threshold, 0.5);
            PIPEPP_OPTION(holl_fill_num_dilate, 0u, u8"지정한 횟수만큼 dilate-erode 연산을 반복 적용", pipepp::verify::maximum(50u));
            PIPEPP_OPTION(holl_fill_num_erode, 0u, u8"지정한 횟수만큼 dilate-erode 연산을 반복 적용", pipepp::verify::maximum(50u));
        };
    };

    struct input_type {
        cv::Mat3b debug;

        // 원본 이미지를 특정 색공간으로 변환한 도메인입니다.
        // marker::filter::method == 0일 때는 color range filter를 계산하는 도메인입니다.
        //
        cv::Mat3b domain;
        cv::Mat3b conv_domain;

        cv::Mat1b lightness;     // marker::filter::method == 1일때만 값을 지정하는 밝기 채널입니다.
        cv::Mat1b all_none_blue; //

        imgproc::img_t const* p_imdesc;
        cv::Vec3f             init_table_pos;
        cv::Vec3f             init_table_rot;

        std::span<const cv::Vec2f> contour;
        std::vector<cv::Vec3f>     marker_model;
    };
    struct output_type {
        cv::Mat1f marker_weight_map;
    };

    pipepp::pipe_error operator()(pipepp::execution_context& ec, input_type const& in, output_type& out);
    static void        link(shared_data & sd, pipepp::execution_context & ec,
                            input_type & i, pipepp::detail::option_base const& opt);

public:
    table_marker_finder();
    ~table_marker_finder();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};


struct marker_solver_cpu {
    PIPEPP_DECLARE_OPTION_CLASS(marker_solver_cpu);
    PIPEPP_OPTION_AUTO(enable_debug_glyphs, true, "debug");
    PIPEPP_OPTION_AUTO(enable_debug_mats, true, "debug");

    struct solver {
        PIPEPP_DECLARE_OPTION_CATEGORY("Solver");

        PIPEPP_OPTION(num_iter, 5);
        PIPEPP_OPTION(error_base, 1.14);
        PIPEPP_OPTION(variant_rot, 0.1);
        PIPEPP_OPTION(variant_pos, 0.1);
        PIPEPP_OPTION(variant_rot_axis, 0.005);
        PIPEPP_OPTION(narrow_rate_pos, 0.5);
        PIPEPP_OPTION(narrow_rate_rot, 0.5);
        PIPEPP_OPTION(num_cands, 600);
        PIPEPP_OPTION(do_parallel, true);
        PIPEPP_OPTION(confidence_amp, 1.5);
        PIPEPP_OPTION(min_valid_marker_size, 1.2);
    };

    struct input_type {
        imgproc::img_t const* img_ptr;
        cv::Size              img_size;

        cv::Vec3f table_pos_init;
        cv::Vec3f table_rot_init;

        cv::Mat const*                debug_mat;
        std::vector<cv::Vec2f> const* p_table_contour;

        cv::UMat const* u_hsv;

        cv::Vec2f FOV_degree;

        std::vector<cv::Vec3f> marker_model;
        std::vector<cv::Vec2f> markers;
        std::vector<float>     weights;
    };

    struct output_type {
        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
        float     confidence;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void        output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);

private:
};

/**
 * GPU를 활용하는 solver입니다. GPU에서 candidate를 생성하고 투영하여, tvec 및 rvec, suitability sum 리스트를 생성합니다.
 */
PIPEPP_EXECUTOR(marker_solver_gpu)
{
    static constexpr int TILE_LONG_SIZE = 1024;
    static constexpr int TILE_SIZE      = 32;

    PIPEPP_CATEGORY(debug, "Debug"){
        PIPEPP_OPTION(draw_position_trace, false);
        PIPEPP_OPTION(draw_result, true);
    };

    PIPEPP_CATEGORY(solve, "Solve")
    {
        PIPEPP_OPTION(num_iteration, 5u, u8"Iteration 횟수입니다. 부하를 선형적으로 증가시킵니다.");
        PIPEPP_OPTION(num_location_cands, 324u, u8"Iteration 당 고려할 후보의 개수입니다. 자동으로 타일 단위로 잘립니다.");
        PIPEPP_OPTION(num_rotation_cands, 128u, u8"Iteration당 고려할 회전의 개수입니다. 자동으로 타일 단위로 자릅니다.");
        PIPEPP_OPTION(var_axis, 0.001, u8"각 후보가 시작 위치에서 얼마만큼 축을 변동할지 결정합니다.");
        PIPEPP_OPTION(var_rotation_deg, 12.0, u8"각 후보의 회전축의 norm 변동 폭을 결정합니다. degree 단위");
        PIPEPP_OPTION(var_location, 0.1, u8"각 후보의 이동량의 변동 폭을 결정합니다.");
        PIPEPP_OPTION(rotation_narrow_rate, 0.4, u8"각 iteration당, 감소시킬 회전 폭입니다.");
        PIPEPP_OPTION(location_narrow_rate, 0.4, u8"각 iteration당, 감소시킬 위치 폭입니다.");
        PIPEPP_OPTION(axis_narrow_rate, 0.4, u8"각 iteration당, 감소시킬 위치 폭입니다.");

        PIPEPP_OPTION(suitability_threshold, 0.1, u8"iteration을 이어갈 최소한의 suitability");
    };

    struct input_type {
        cv::Mat1f marker_weight_map;
        cv::Vec3f init_local_table_pos;
        cv::Vec3f init_local_table_rot;

        imgproc::img_t const* p_imdesc;
        cv::Mat3b             debug_mat;

        std::vector<cv::Vec3f> marker_model;
    };

    struct output_type {
        cv::Vec3f local_table_pos;
        cv::Vec3f local_table_rot;
        float     confidence;
    };

    void        operator()(pipepp::execution_context& ec, input_type const& i, output_type& o);
    static void link(shared_data & sd, table_marker_finder::output_type const& o, input_type& i);

public:
    marker_solver_gpu();
    ~marker_solver_gpu();

private:
    struct impl_type;
    std::unique_ptr<impl_type> _m;
};
} // namespace billiards::pipes
