#pragma once
#include "recognizer.hpp"
#include "../image_processing.hpp"

namespace billiards::pipes
{
using namespace std::literals;

namespace helpers
{
struct table_edge_extender {
    // inputs
    cv::Vec3f table_rot, table_pos;
    imgproc::img_t const* p_imdesc;
    std::span<const cv::Vec2f> table_contour;

    // opts
    int num_insert_contour_vertexes = 5;
    double table_border_range_outer = 0.06;
    double table_border_range_inner = 0.06;

    cv::Mat const* debug_mat = {};

    void operator()(pipepp::execution_context& ec, cv::Mat& marker_area_mask);
};

struct kernel_visualizer {
    std::span<cv::Vec3f> vtxs;
    int kernel_view_size = 200;
    size_t positive_index_fence = 0;

    cv::Mat3b operator()(pipepp::execution_context& ec)
    {
        PIPEPP_REGISTER_CONTEXT(ec);

        PIPEPP_ELAPSE_SCOPE("Kernel visualization");
        auto scale = kernel_view_size;
        auto mult = scale / 4;
        auto radius = std::max(1, scale / 100);
        cv::Mat3b kernel_view(scale, scale, {0, 0, 0});
        cv::Point center(scale / 2, scale / 2);
        cv::Scalar colors[] = {{0, 255, 0}, {0, 0, 255}};

        for (auto idx : kangsw::counter(vtxs.size())) {
            auto vtx = vtxs[idx];
            cv::Point pt(vtx[0] * mult, -vtx[2] * mult);
            cv::circle(kernel_view, center + pt, radius, colors[idx >= positive_index_fence]);
        }

        return kernel_view;
    }
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
        PIPEPP_OPTION(kernel_view_size, 200);
        PIPEPP_OPTION(current_kernel_view_scale, 0.05f);
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
        PIPEPP_OPTION(random_seed, 42);
    };

    PIPEPP_CATEGORY(marker, "Marker Search")
    {
        PIPEPP_OPTION(radius, 0.005, "Units in centimeters");

        PIPEPP_CATEGORY(pp, "Preprocessing")
        {
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
        };

        PIPEPP_CATEGORY(filter, "Filtering")
        {
            PIPEPP_OPTION(method, 0,
                          "[0] Simple color range filter \n"
                          "[1] Lightness edge: L of Lab \n"
                          "[2] Lightness edge: Y of YUV \n"
                          "[3] Lightness edge: V of HSV \n",
                          pipepp::verify::contains(0, 1, 2, 3));

            PIPEPP_OPTION(color_space, "HSV"s, u8"마커의 필터를 적용할 색공간입니다.", verify::color_space_string_verify);
            PIPEPP_OPTION(pivot_color, cv::Vec3b(233, 233, 233), u8"마커의 대표 색상입니다. 색 공간에 의존적입니다.");

            PIPEPP_OPTION(method_0_range_lo, cv::Vec3b(125, 125, 125));
            PIPEPP_OPTION(method_0_range_hi, cv::Vec3b(255, 255, 255));

            PIPEPP_OPTION(method_1_threshold, 0.5);
            PIPEPP_OPTION(method_1_hole_filling_cnt, 0, u8"지정한 횟수만큼 dilate-erode 연산을 반복 적용");
        };
    };

    struct input_type {
        cv::Mat3b debug;

        // 원본 이미지를 특정 색공간으로 변환한 도메인입니다.
        // marker::filter::method == 0일 때는 color range filter를 계산하는 도메인입니다.
        //
        cv::Mat3b domain;

        cv::Mat1b lightness; // marker::filter::method == 1일때만 값을 지정하는 밝기 채널입니다.

        imgproc::img_t const* p_imdesc;
        cv::Vec3f init_table_pos;
        cv::Vec3f init_table_rot;

        std::span<const cv::Vec2f> contour;
        std::vector<cv::Vec3f> marker_model;
    };
    struct output_type {
        cv::Mat1f marker_weight_map;
    };

    pipepp::pipe_error operator()(pipepp::execution_context& ec, input_type const& in, output_type& out);

    static bool link(shared_data & sd, input_type & i, pipepp::detail::option_base const& opt)
    {
        {
            auto _lck = sd.state->lock();
            i.init_table_pos = sd.state->table.pos;
            i.init_table_rot = sd.state->table.rot;
        }

        i.debug = sd.debug_mat;
        i.p_imdesc = &sd.imdesc_bkup;

        i.contour = sd.table.contour;
        sd.get_marker_points_model(i.marker_model);

        i.domain = sd.retrieve_image_in_colorspace(marker::filter::color_space(opt));

        auto method = marker::filter::method(opt);

        if (method > 0) {
            cv::Mat split[3];
            cv::split(i.domain, split);

            switch (method) {
                case 1: [[fallthrough]];
                case 2: i.lightness = split[0]; break;
                case 3: i.lightness = split[2]; break;

                default:
                    return false;
            }
        } else {
            i.lightness = {};
        }

        return i.contour.empty() == false;
    }

public:
    table_marker_finder();
    ~table_marker_finder();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
} // namespace billiards::pipes
