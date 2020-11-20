#pragma once
#include <opencv2/core/matx.hpp>

#include "shared_data.hpp"
#include "../image_processing.hpp"
#include "pipepp/pipeline.hpp"

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
    PIPEPP_DEFINE_OPTION_2(table_size_outer, cv::Vec2d(), "table");
    PIPEPP_DEFINE_OPTION_2(table_size_inner, cv::Vec2d(), "table");
    PIPEPP_DEFINE_OPTION_2(table_size_fit, cv::Vec2d(), "table");
    PIPEPP_DEFINE_OPTION_2(table_filter_alpha_pos, 0.3, "table");
    PIPEPP_DEFINE_OPTION_2(table_filter_alpha_rot, 0.3, "table");
    PIPEPP_DEFINE_OPTION_2(table_filter_jump_threshold_distance, 0.1, "table");
    PIPEPP_DEFINE_OPTION_2(camera_FOV, cv::Vec2d(88, 58), "common");

    // data
    std::shared_ptr<shared_state> state;

    recognizer_t::parameter_type param_bkup;
    recognizer_t::process_finish_callback_type callback;

    cv::Mat debug_mat;

    cv::Mat rgb, hsv;
    cv::UMat u_rgb, u_hsv;

    struct {
        std::vector<cv::Vec2f> contour;

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

    PIPEPP_DEFINE_OPTION_2(debug_show_0_filtered, false, "debug");
    PIPEPP_DEFINE_OPTION_2(debug_show_1_edge, false, "debug");

    PIPEPP_DEFINE_OPTION_2(area_threshold_ratio, 0.2, "contour", "Minimul pixel size of table candidate area");
    PIPEPP_DEFINE_OPTION_2(approx_epsilon_preprocess, 5.0, "contour", "Epsilon value used for approximate table contours");
    PIPEPP_DEFINE_OPTION_2(approx_epsilon_convex_hull, 1.0, "contour", "Epsilon value used for approximate table contours after convex hull operation.");

    struct input_type {
        cv::Mat debug_display;
        cv::UMat u_hsv;
    };
    struct output_type {
        std::vector<cv::Vec2f> table_contour_candidate;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const&, input_resize::output_type const&, input_type&);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct table_edge_solver {
    PIPEPP_DEFINE_OPTION_CLASS(table_edge_solver);

    PIPEPP_DEFINE_OPTION_2(pnp_error_exp_fn_base,
                           1.06,
                           "PnP",
                           "Base of exponent function which is applied to calculate confidence of full-PNP solver result");
    PIPEPP_DEFINE_OPTION_2(pnp_conf_threshold,
                           1.06,
                           "PnP",
                           "Minimum required confidence value of full-PNP solver.");

    PIPEPP_DEFINE_OPTION_2(debug_show_partial_glyphs, true, "debug");

    PIPEPP_DEFINE_OPTION_2(enable_partial_solver, true, "flags");

    PIPEPP_DEFINE_OPTION_2(enable_partial_parallel_solve, true, "partial");
    PIPEPP_DEFINE_OPTION_2(partial_solver_iteration, 10, "partial solver");
    PIPEPP_DEFINE_OPTION_2(partial_solver_candidates, 265, "partial solver");
    PIPEPP_DEFINE_OPTION_2(rotation_axis_variant, 0.015f, "partial solver");
    PIPEPP_DEFINE_OPTION_2(rotation_amount_variant, 0.015f, "partial solver");
    PIPEPP_DEFINE_OPTION_2(distance_variant, 0.01f, "partial solver");
    PIPEPP_DEFINE_OPTION_2(border_margin, 3, "partial solver");
    PIPEPP_DEFINE_OPTION_2(iteration_narrow_rate, 0.8f, "partial solver");
    PIPEPP_DEFINE_OPTION_2(error_function_base, 1.06, "partial");

    PIPEPP_DEFINE_OPTION_2(cull_window_top_left, cv::Vec2d(0, 0), "partial cull");
    PIPEPP_DEFINE_OPTION_2(cull_window_bottom_right, cv::Vec2d(1, 1), "partial cull");

    PIPEPP_DEFINE_OPTION_2(apply_weight, 0.3, "partial");

    struct input_type {
        cv::Vec2f FOV_degree;

        cv::Mat debug_mat;
        recognizer_t::parameter_type const* img_ptr;

        cv::Size img_size;

        std::vector<cv::Vec2f> const* table_contour;
        cv::Vec2f table_fit_size;

        cv::Vec3f table_pos_init;
        cv::Vec3f table_rot_init;
    };

    struct output_type {
        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
        float confidence;
        bool can_jump;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const& sd, contour_candidate_search::output_type const& i, input_type& o);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

} // namespace billiards::pipes