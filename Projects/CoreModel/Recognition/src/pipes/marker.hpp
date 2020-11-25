#pragma once
#include "recognizer.hpp"

namespace billiards::pipes
{
/**
 * TODO
 * ��Ŀ�� Ž���մϴ�. ��� �� ���� ����, Sparse Kernel�� ������ ã�Ƴ��ϴ�.
 * �� ��, Ŀ���� �⺻���� ������ �� ����� 3D �������� ��ȯ�ϰ�, �� ���� ���ؽ� ���̴��� ������ ����ϴ�.
 *
 * @details
 *
 * ��� Ŀ�� ������ �� ���ؽ��� X, Z ���(���̺�� ���� ���)�� �����մϴ�. ���̺��� ī�޶� ���� ��� �����̼����� �� ���ؽ��� ȸ����Ű�� ȭ�鿡 ���� �����ϸ�, ��� Ŀ���� ȹ���� �� �ֽ��ϴ�.
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