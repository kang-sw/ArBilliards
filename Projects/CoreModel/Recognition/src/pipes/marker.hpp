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
        PIPEPP_OPTION(kernel_view_size, 200);
        PIPEPP_OPTION(current_kernel_view_scale, 0.05f);
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
        PIPEPP_OPTION(random_seed, 42);
    };

    PIPEPP_CATEGORY(marker, "Marker Search")
    {
        PIPEPP_OPTION(radius, 0.005, "Units in centimeters");

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
                          "[1] Lightness edge: L of Lab \n"
                          "[2] Lightness edge: Y of YUV \n"
                          "[3] Lightness edge: V of HSV \n",
                          pipepp::verify::contains(0, 1, 2, 3));

            PIPEPP_OPTION(color_space, "HSV"s, u8"��Ŀ�� ���͸� ������ �������Դϴ�.", verify::color_space_string_verify);
            PIPEPP_OPTION(pivot_color, cv::Vec3b(233, 233, 233), u8"��Ŀ�� ��ǥ �����Դϴ�. �� ������ �������Դϴ�.");

            PIPEPP_OPTION(method_0_range_lo, cv::Vec3b(125, 125, 125));
            PIPEPP_OPTION(method_0_range_hi, cv::Vec3b(255, 255, 255));

            PIPEPP_OPTION(method_1_threshold, 0.5);
            PIPEPP_OPTION(method_1_hole_filling_cnt, 0, u8"������ Ƚ����ŭ dilate-erode ������ �ݺ� ����");
        };
    };

    struct input_type {
        cv::Mat3b debug;

        // ���� �̹����� Ư�� ���������� ��ȯ�� �������Դϴ�.
        // marker::filter::method == 0�� ���� color range filter�� ����ϴ� �������Դϴ�.
        //
        cv::Mat3b domain;

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
