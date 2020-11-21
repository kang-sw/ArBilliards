#pragma once
#include <opencv2/core/matx.hpp>

#include "../image_processing.hpp"
#include "kangsw/spinlock.hxx"
#include "pipepp/execution_context.hpp"
#include "pipepp/pipeline.hpp"

namespace billiards::pipes
{
auto build_pipe() -> std::shared_ptr<pipepp::pipeline<struct shared_data, struct input_resize>>;
}

namespace billiards::pipes
{
struct shared_state {
    auto lock() const { return std::unique_lock{lock_}; }

    struct {
        cv::Vec3f pos = {}, rot = cv::Vec3f(1, 0, 0);
    } table;

private:
    mutable kangsw::spinlock lock_;
};

struct shared_data : pipepp::base_shared_context {
    // options
    PIPEPP_DECLARE_OPTION_CLASS(shared_data);
    PIPEPP_OPTION(table_size_outer, cv::Vec2d(), "table");
    PIPEPP_OPTION(table_size_inner, cv::Vec2d(), "table");
    PIPEPP_OPTION(table_size_fit, cv::Vec2d(), "table");
    PIPEPP_OPTION(table_filter_alpha_pos, 0.3, "table");
    PIPEPP_OPTION(table_filter_alpha_rot, 0.3, "table");
    PIPEPP_OPTION(table_filter_jump_threshold_distance, 0.1, "table");
    PIPEPP_OPTION(camera_FOV, cv::Vec2d(88, 58), "common");

    // data
    std::shared_ptr<shared_state> state;

    recognizer_t::frame_desc param_bkup;
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
    PIPEPP_DECLARE_OPTION_CLASS(input_resize);
    PIPEPP_OPTION(desired_image_width, 1280);

    PIPEPP_OPTION(debug_show_source, false);
    PIPEPP_OPTION(debug_show_hsv, false);

    using input_type = recognizer_t::frame_desc;
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
    PIPEPP_DECLARE_OPTION_CLASS(contour_candidate_search);
    PIPEPP_OPTION(table_color_filter_0_lo, cv::Vec3b(0, 0, 0));
    PIPEPP_OPTION(table_color_filter_1_hi, cv::Vec3b(180, 255, 255));

    PIPEPP_OPTION(show_0_filtered, false, "debug");
    PIPEPP_OPTION(show_1_edge, false, "debug");

    PIPEPP_OPTION(area_threshold_ratio, 0.2, "contour", "Minimul pixel size of table candidate area");
    PIPEPP_OPTION(approx_epsilon_preprocess, 5.0, "contour", "Epsilon value used for approximate table contours");
    PIPEPP_OPTION(approx_epsilon_convex_hull, 1.0, "contour", "Epsilon value used for approximate table contours after convex hull operation.");

    struct preprocess {
        PIPEPP_OPTION(num_erode_prev, 0, "preprocess", "Number of erode operation before apply dilate. Dilate count determined automatically. ");
        PIPEPP_OPTION(num_erode_post, 0, "preprocess", "Number of dilate operation after apply dilate. Dilate count is determined automatically.");
    };

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
    PIPEPP_DECLARE_OPTION_CLASS(table_edge_solver);

    PIPEPP_OPTION(pnp_error_exp_fn_base,
                  1.06,
                  "PnP",
                  "Base of exponent function which is applied to calculate confidence of full-PNP solver result");
    PIPEPP_OPTION(pnp_conf_threshold,
                  1.06,
                  "PnP",
                  "Minimum required confidence value of full-PNP solver.");

    PIPEPP_OPTION(debug_show_partial_glyphs, true, "debug");

    PIPEPP_OPTION(enable_partial_solver, true, "flags");

    PIPEPP_OPTION(enable_partial_parallel_solve, true, "partial");
    PIPEPP_OPTION(partial_solver_iteration, 10, "partial solver");
    PIPEPP_OPTION(partial_solver_candidates, 265, "partial solver");
    PIPEPP_OPTION(rotation_axis_variant, 0.015f, "partial solver");
    PIPEPP_OPTION(rotation_amount_variant, 0.015f, "partial solver");
    PIPEPP_OPTION(distance_variant, 0.01f, "partial solver");
    PIPEPP_OPTION(border_margin, 3, "partial solver");
    PIPEPP_OPTION(iteration_narrow_rate, 0.8f, "partial solver");
    PIPEPP_OPTION(error_function_base, 1.06, "partial");

    PIPEPP_OPTION(cull_window_top_left, cv::Vec2d(0, 0), "partial cull");
    PIPEPP_OPTION(cull_window_bottom_right, cv::Vec2d(1, 1), "partial cull");

    PIPEPP_OPTION(apply_weight, 0.3, "partial");

    struct input_type {
        cv::Vec2f FOV_degree;

        cv::Mat debug_mat;
        recognizer_t::frame_desc const* img_ptr;

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

struct marker_solver {
    PIPEPP_DECLARE_OPTION_CLASS(marker_solver);
    PIPEPP_OPTION(enable_debug_glyphs, true, "debug");
    PIPEPP_OPTION(enable_debug_mats, true, "debug");
    PIPEPP_OPTION(show_marker_area_mask, false, "debug");

    PIPEPP_OPTION(num_insert_contour_vtx, 5, "PP 0: marker area");
    PIPEPP_OPTION(table_border_range_outer, 0.06, "PP 0: marker area");
    PIPEPP_OPTION(table_border_range_inner, 0.02, "PP 0: marker area");

    PIPEPP_OPTION(laplacian_mask_threshold, 0.1, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_min_rad, 0.5, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_max_rad, 10.0, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_min_size, 1, "PP 1: filtering");

    struct marker {
        PIPEPP_OPTION(count_x, 9, "marker");
        PIPEPP_OPTION(count_y, 5, "marker");
        PIPEPP_OPTION(felt_width, 1.735f, "marker");
        PIPEPP_OPTION(felt_height, 0.915f, "marker");
        PIPEPP_OPTION(dist_from_felt_long, 0.012f, "marker");
        PIPEPP_OPTION(dist_from_felt_short, 0.012f, "marker");
        PIPEPP_OPTION(step, 0.206f, "marker");
        PIPEPP_OPTION(width_shift_a, 0.0f, "marker");
        PIPEPP_OPTION(width_shift_b, 0.0f, "marker");
        PIPEPP_OPTION(height_shift_a, 0.0f, "marker");
        PIPEPP_OPTION(height_shift_b, 0.01f, "marker");
    };

    struct solver {
        PIPEPP_OPTION(iteration, 5, "Solver");

        PIPEPP_OPTION(error_base, 1.14, "Solver");
        PIPEPP_OPTION(variant_rot, 0.1, "Solver");
        PIPEPP_OPTION(variant_pos, 0.1, "Solver");
        PIPEPP_OPTION(variant_rot_axis, 0.005, "Solver");
        PIPEPP_OPTION(narrow_rate_pos, 0.5, "Solver");
        PIPEPP_OPTION(narrow_rate_rot, 0.5, "Solver");
        PIPEPP_OPTION(num_cands, 600, "Solver");
        PIPEPP_OPTION(num_iter, 5, "Solver");
        PIPEPP_OPTION(do_parallel, true, "Solver");
        PIPEPP_OPTION(confidence_amp, 1.5, "Solver");
        PIPEPP_OPTION(min_valid_marker_size, 1.2, "Solver");
    };

    struct input_type {
        imgproc::img_t const* img_ptr;
        cv::Size img_size;

        cv::Vec3f table_pos_init;
        cv::Vec3f table_rot_init;

        cv::Mat const* debug_mat;
        std::vector<cv::Vec2f> const* table_contour;

        cv::UMat const* u_hsv;

        cv::Vec2f FOV_degree;
    };

    struct output_type {
        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
        float confidence;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const& sd, table_edge_solver::output_type const& i, input_type& o);

    static void get_marker_points_model(pipepp::execution_context& ec, std::vector<cv::Vec3f>& model);
};

} // namespace billiards::pipes