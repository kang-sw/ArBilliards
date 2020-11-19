#pragma once
#include <opencv2/core/matx.hpp>

#include "shared_data.hpp"
#include "../recognition.hpp"
#include "pipepp/pipeline.hpp"

namespace billiards::imgproc
{
void filter_hsv(cv::InputArray input, cv::OutputArray output, cv::Vec3f min_hsv, cv::Vec3f max_hsv);
bool is_border_pixel(cv::Rect img_size, cv::Vec2i pixel, int margin = 3);
} // namespace billiards::imgproc

namespace billiards::pipes
{
auto build_pipe() -> std::shared_ptr<pipepp::pipeline<struct shared_data, struct input_resize>>;
}

namespace billiards::pipes
{
struct shared_state {
    struct {
        cv::Vec3f pos = {}, rot = cv::Vec3f(1, 0, 0);
    } table;
};

struct shared_data : pipepp::base_shared_context {
    // options
    PIPEPP_DEFINE_OPTION_CLASS(shared_data);
    PIPEPP_DEFINE_OPTION_2(table_size_outer, cv::Vec2d());
    PIPEPP_DEFINE_OPTION_2(table_size_inner, cv::Vec2d());
    PIPEPP_DEFINE_OPTION_2(table_size_fit, cv::Vec2d());

    // data
    std::shared_ptr<shared_state> state;

    recognizer_t::parameter_type param_bkup;
    recognizer_t::process_finish_callback_type callback;

    cv::Mat debug_mat;

    cv::Mat rgb, hsv;
    cv::UMat u_rgb, u_hsv;

    struct {
        cv::Vec3f pos, rot;
        float confidence;
    } table;
};

struct input_resize {
    PIPEPP_DEFINE_OPTION_CLASS(input_resize);
    PIPEPP_DEFINE_OPTION_2(desired_image_width, 1280);

    PIPEPP_DEFINE_OPTION_2(debug_show_source, false);
    PIPEPP_DEFINE_OPTION_2(debug_show_hsv, false);

    using input_type = recognizer_t::parameter_type;
    struct output_type {
        cv::Size img_size;
        cv::Mat rgb;
        cv::Mat hsv;
        cv::UMat u_rgb;
        cv::UMat u_hsv;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct contour_candidate_search {
    PIPEPP_DEFINE_OPTION_CLASS(contour_candidate_search);
    PIPEPP_DEFINE_OPTION_2(table_color_filter_0_lo, cv::Vec3b(0, 0, 0));
    PIPEPP_DEFINE_OPTION_2(table_color_filter_1_hi, cv::Vec3b(180, 255, 255));

    PIPEPP_DEFINE_OPTION_2(debug_show_0_filtered, false);
    PIPEPP_DEFINE_OPTION_2(debug_show_1_edge, false);

    PIPEPP_DEFINE_OPTION_2(contour_area_threshold_ratio, 0.2);
    PIPEPP_DEFINE_OPTION_2(contour_approx_epsilon_preprocess, 5.0);
    PIPEPP_DEFINE_OPTION_2(contour_approx_epsilon_convex_hull, 1.0);

    struct input_type {
        cv::Mat debug_display;
        cv::UMat u_hsv;
    };
    struct output_type {
        std::vector<cv::Vec2f> table_contour_candidate;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const&, input_resize::output_type const&, input_type&);
};

struct table_edge_solver {
    PIPEPP_DEFINE_OPTION_CLASS(table_edge_solver);

    struct input_type {
        cv::Mat debug_display;
        recognizer_t::parameter_type const* img_ptr;

        std::vector<cv::Vec2f>* table_contours;
        cv::Vec2f table_fit_size;

        cv::Vec3f table_pos_init;
        cv::Vec3f table_rot_init;
    };

    struct output_type {
        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
        float confidence;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const&, input_resize::output_type const&, input_type&);
};

} // namespace billiards::pipes