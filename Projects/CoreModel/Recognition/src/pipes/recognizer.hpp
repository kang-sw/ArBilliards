#pragma once
#include <opencv2/core/matx.hpp>

#include "../image_processing.hpp"
#include "kangsw/spinlock.hxx"
#include "pipepp/execution_context.hpp"
#include "pipepp/pipeline.hpp"

namespace pipepp
{
namespace impl__
{
class option_base;
}
} // namespace pipepp

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
    struct table {
        struct size {
            PIPEPP_OPTION(outer, cv::Vec2d(1.8, 0.98), "table");
            PIPEPP_OPTION(innter, cv::Vec2d(1.653, 0.823), "table");
            PIPEPP_OPTION(fit, cv::Vec2d(1.735, 0.915), "table");
        };
        struct filter {
            PIPEPP_OPTION(alpha_pos, 0.3, "table");
            PIPEPP_OPTION(alpha_rot, 0.3, "table");
            PIPEPP_OPTION(jump_threshold_distance, 0.1, "table");
        };
    };

    PIPEPP_OPTION(camera_FOV, cv::Vec2d(84.855, 53.27), "common");

    struct ball {
        PIPEPP_OPTION(radius, 0.030239439175, "ball");
    };

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
    PIPEPP_OPTION(table_color_filter_0_lo, cv::Vec3b(175, 150, 60));
    PIPEPP_OPTION(table_color_filter_1_hi, cv::Vec3b(10, 255, 255));

    PIPEPP_OPTION(show_0_filtered, false, "debug");
    PIPEPP_OPTION(show_1_edge, false, "debug");

    PIPEPP_OPTION(area_threshold_ratio, 0.1, "contour", "Minimul pixel size of table candidate area");
    PIPEPP_OPTION(approx_epsilon_preprocess, 5.0, "contour", "Epsilon value used for approximate table contours");
    PIPEPP_OPTION(approx_epsilon_convex_hull, 1.0, "contour", "Epsilon value used for approximate table contours after convex hull operation.");

    struct preprocess {
        PIPEPP_OPTION(num_erode_prev, 2, "preprocess", "Number of erode operation before apply dilate. Dilate count determined automatically. ");
        PIPEPP_OPTION(num_erode_post, 6, "preprocess", "Number of dilate operation after apply dilate. Dilate count is determined automatically.");
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
    PIPEPP_OPTION(enable_partial_parallel_solve, true, "flags");

    struct partial {
        struct solver {
            PIPEPP_OPTION(iteration, 5, "partial.solver");
            PIPEPP_OPTION(candidates, 266, "partial.solver");
            PIPEPP_OPTION(rotation_axis_variant, 0.008f, "partial.solver");
            PIPEPP_OPTION(rotation_amount_variant, 0.007f, "partial.solver");
            PIPEPP_OPTION(distance_variant, 0.12f, "partial.solver");
            PIPEPP_OPTION(border_margin, 3, "partial.solver");
            PIPEPP_OPTION(iteration_narrow_rate, 0.5f, "partial.solver");
            PIPEPP_OPTION(error_function_base, 1.06, "partial.solver");
        };

        PIPEPP_OPTION(cull_window_top_left, cv::Vec2d(0.05, 0.05), "partial.cull");
        PIPEPP_OPTION(cull_window_bottom_right, cv::Vec2d(0.95, 0.95), "partial.cull");

        PIPEPP_OPTION(apply_weight, 0.3, "partial");
    };

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

    PIPEPP_OPTION(num_insert_contour_vtx, 5, "PP 0. Marker area");
    PIPEPP_OPTION(table_border_range_outer, 0.06, "PP 0. Marker area");
    PIPEPP_OPTION(table_border_range_inner, 0.02, "PP 0. Marker area");

    PIPEPP_OPTION(laplacian_mask_threshold, 0.5, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_min_rad, 0.5, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_max_rad, 10.0, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_min_size, 1, "PP 1: filtering");

    struct marker {
        PIPEPP_DECLARE_OPTION_CATEGORY("marker");

        PIPEPP_OPTION_CAT(count_x, 9);
        PIPEPP_OPTION_CAT(count_y, 5);
        PIPEPP_OPTION_CAT(felt_width, 1.735f);
        PIPEPP_OPTION_CAT(felt_height, 0.915f);
        PIPEPP_OPTION_CAT(dist_from_felt_long, 0.012f);
        PIPEPP_OPTION_CAT(dist_from_felt_short, 0.012f);
        PIPEPP_OPTION_CAT(step, 0.206f);
        PIPEPP_OPTION_CAT(width_shift_a, 0.0f);
        PIPEPP_OPTION_CAT(width_shift_b, 0.0f);
        PIPEPP_OPTION_CAT(height_shift_a, 0.0f);
        PIPEPP_OPTION_CAT(height_shift_b, 0.01f);
    };

    struct solver {
        PIPEPP_DECLARE_OPTION_CATEGORY("Solver");

        PIPEPP_OPTION_CAT(iteration, 5);
        PIPEPP_OPTION_CAT(error_base, 1.14);
        PIPEPP_OPTION_CAT(variant_rot, 0.1);
        PIPEPP_OPTION_CAT(variant_pos, 0.1);
        PIPEPP_OPTION_CAT(variant_rot_axis, 0.005);
        PIPEPP_OPTION_CAT(narrow_rate_pos, 0.5);
        PIPEPP_OPTION_CAT(narrow_rate_rot, 0.5);
        PIPEPP_OPTION_CAT(num_cands, 600);
        PIPEPP_OPTION_CAT(num_iter, 5);
        PIPEPP_OPTION_CAT(do_parallel, true);
        PIPEPP_OPTION_CAT(confidence_amp, 1.5);
        PIPEPP_OPTION_CAT(min_valid_marker_size, 1.2);
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
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);

    static void get_marker_points_model(pipepp::execution_context& ec, std::vector<cv::Vec3f>& model);
};

struct ball_search {
    PIPEPP_DECLARE_OPTION_CLASS(ball_search);
    PIPEPP_OPTION(show_debug_mat, false, "debug");

    struct field {
        inline static const std::string _category = "Step 0: Field.";

        struct red {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "Red");
            PIPEPP_OPTION_CAT(color, cv::Vec2f{}, "Representative Hue, Saturation color");
            PIPEPP_OPTION_CAT(weight_hs, cv::Vec2f{}, "Weight per channel");
            PIPEPP_OPTION_CAT(error_fn_base, 1.04, "Weight per channel");
        };

        struct orange {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "Orange");
            PIPEPP_OPTION_CAT(color, cv::Vec2f{}, "Representative Hue, Saturation color");
            PIPEPP_OPTION_CAT(weight_hs, cv::Vec2f{}, "Weight per channel");
            PIPEPP_OPTION_CAT(error_fn_base, 1.04, "Weight per channel");
        };

        struct white {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "White");
            PIPEPP_OPTION_CAT(color, cv::Vec2f{}, "Representative Hue, Saturation color");
            PIPEPP_OPTION_CAT(weight_hs, cv::Vec2f{}, "Weight per channel");
            PIPEPP_OPTION_CAT(error_fn_base, 1.04, "Weight per channel");
        };
    };

    struct random_sample {
    };

    struct matching {
        PIPEPP_OPTION(cushion_center_gap, 0.0, "2.matching");
    };

    struct input_type {
        pipepp::impl__::option_base const* opt_shared;
        imgproc::img_t const* imdesc;
        cv::Mat const* debug_mat;

        cv::UMat u_rgb, u_hsv;

        cv::Size img_size;
        std::vector<cv::Vec2f> const* table_contour;

        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
    };
    struct output_type {
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const& sd, marker_solver::output_type const& i, input_type& o);
};

} // namespace billiards::pipes