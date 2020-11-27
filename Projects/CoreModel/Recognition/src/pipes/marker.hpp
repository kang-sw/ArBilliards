#pragma once
#include "../image_processing.hpp"
#include "recognizer.hpp"

namespace billiards::pipes {
using namespace std::literals;

namespace helpers {
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

    struct {
        std::vector<cv::Vec2f> outer_contour;
        std::vector<cv::Vec2f> inner_contour;
    } output;

    void operator()(pipepp::execution_context& ec, cv::Mat& marker_area_mask);
};

} // namespace helpers

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
                      u8"�߽������κ���, ���� ����ġ�� �򰡵Ǵ� �����Դϴ�.",
                      pipepp::verify::clamp_all<cv::Vec2d>(0, 1) | pipepp::verify::ascending<cv::Vec2d>());
        PIPEPP_OPTION(negative_area,
                      cv::Vec2d(1, 2),
                      u8"�߽������κ���, ���� ����ġ�� �򰡵Ǵ� �����Դϴ�.",
                      pipepp::verify::minimum_all<cv::Vec2d>(0) | pipepp::verify::ascending<cv::Vec2d>());
        PIPEPP_OPTION(generator_positive_radius, 10u, "", pipepp::verify::maximum(10000u));
        PIPEPP_OPTION(generator_negative_radius, 10u, "", pipepp::verify::maximum(10000u));
        PIPEPP_OPTION(random_seed, 42u);
    };

    PIPEPP_OPTION(marker_radius, 0.005, "Units in centimeters");

    PIPEPP_CATEGORY(pp, "Preprocessing")
    {
        PIPEPP_OPTION(num_inserted_contours, 5,
                      u8"��Ŀ Ž���� ���� �籸�� �� ������ Ȯ���� ��, "
                      "���Ӱ� ������ ������ ������ �����Դϴ�. ���� ó�� ��ü�� ��ġ�� ������ �̹��մϴ�.");
        PIPEPP_OPTION(marker_range_outer, 0.1,
                      u8"�Ϲ�������, �籸���� ��Ʈ ��輱���� �ٱ��ʱ����� ���� ���̸� �����մϴ�.\n"
                      "meter ����");
        PIPEPP_OPTION(marker_range_inner, 0.0,
                      u8"�籸���� ��Ʈ ������ �������� ��Ŀ ���� ����ũ�� �����ϴ� �� ����մϴ�.\n"
                      "�Ϲ������� 0�� �����ϸ� ����մϴ�.\n"
                      "meter ����");
    };

    PIPEPP_CATEGORY(filter, "Filtering")
    {
        PIPEPP_OPTION(method, 0,
                      "[0] Simple color range filter \n"
                      "[1] Lightness edge detector\n",
                      pipepp::verify::contains(0, 1));

        PIPEPP_OPTION(filter_color_space, "HSV"s, u8"��ȿ �ȼ��� ������ �������Դϴ�.", verify::color_space_string_verify);
        PIPEPP_OPTION(convolution_color_space, "HSV"s, u8"��Ŀ�� ��������� ������ �������Դϴ�.", verify::color_space_string_verify);

        PIPEPP_CATEGORY(convolution, "Convolution")
        {
            PIPEPP_OPTION(pivot_color, cv::Vec3b(233, 233, 233), u8"��Ŀ�� ��ǥ �����Դϴ�. �� ������ �������Դϴ�.");
            PIPEPP_OPTION(pivot_color_weight, cv::Vec3f(1, 1, 1), u8"��ǥ ������ ���� �����Դϴ�.");
            PIPEPP_OPTION(color_distance_error_base, 1.01f,
                          u8"�Ÿ� ��� �Լ� d(x) = b^(-x)�� ����, b�� ���Դϴ�.",
                          pipepp::verify::minimum(1.f));
            PIPEPP_OPTION(negative_kernel_weight, 1.0f, u8"Negative kernel�� ���� �򰡿� ��ġ�� ���� �����Դϴ�.",
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
            PIPEPP_OPTION(holl_fill_num_dilate, 0u, u8"������ Ƚ����ŭ dilate-erode ������ �ݺ� ����", pipepp::verify::maximum(50u));
            PIPEPP_OPTION(holl_fill_num_erode, 0u, u8"������ Ƚ����ŭ dilate-erode ������ �ݺ� ����", pipepp::verify::maximum(50u));
        };
    };

    struct input_type {
        cv::Mat3b debug;

        // ���� �̹����� Ư�� ���������� ��ȯ�� �������Դϴ�.
        // marker::filter::method == 0�� ���� color range filter�� ����ϴ� �������Դϴ�.
        //
        cv::Mat3b domain;
        cv::Mat3b conv_domain;

        cv::Mat1b lightness; // marker::filter::method == 1�϶��� ���� �����ϴ� ��� ä���Դϴ�.

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
    static void link(shared_data& sd, input_type& i, pipepp::detail::option_base const& opt);

public:
    table_marker_finder();
    ~table_marker_finder();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
} // namespace billiards::pipes
