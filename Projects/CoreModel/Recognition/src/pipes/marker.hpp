#pragma once
#include "recognizer.hpp"

namespace billiards::pipes
{
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
    PIPEPP_CATEGORY(debug, "Debug"){

    };

    PIPEPP_CATEGORY(kernel, "Kernel")
    {
        PIPEPP_OPTION(positive_area,
                      cv::Vec2d(0, 1),
                      "",
                      pipepp::verify::clamp_all<cv::Vec2d>(0, 1) | pipepp::verify::ascending<cv::Vec2d>());
        PIPEPP_OPTION(negative_area,
                      cv::Vec2d(1, 2),
                      "",
                      pipepp::verify::minimum_all<cv::Vec2d>(0) | pipepp::verify::ascending<cv::Vec2d>());
        PIPEPP_OPTION(generator_integral_radius, 10u, "", pipepp::verify::maximum(10000u));
    };

    PIPEPP_CATEGORY(marker, "Marker")
    {
        PIPEPP_OPTION(marker_radius, 0.005, "Units in centimeters");
    };

    struct input_type {
        cv::Mat3b debug;
        cv::Mat source;

        std::vector<cv::Vec2f> contour;
        std::vector<cv::Vec3f> marker_model;
    };
    struct output_type {
        cv::Mat1f marker_weight_map;
    };

    pipepp::pipe_error operator()(pipepp::execution_context& ec, input_type const& in, output_type& out);
    static void link(shared_data const& sd, input_type& i)
    {
        i.contour = sd.table.contour;
        sd.get_marker_points_model(i.marker_model);
    }

public:
    table_marker_finder();
    ~table_marker_finder();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
} // namespace billiards::pipes