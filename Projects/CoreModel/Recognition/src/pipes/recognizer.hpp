#pragma once
#include <opencv2/core/matx.hpp>

#include "../image_processing.hpp"
#include "kangsw/spinlock.hxx"
#include "pipepp/execution_context.hpp"
#include "pipepp/pipeline.hpp"

namespace pipepp
{
namespace detail
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
struct ball_position_desc {
    using clock = std::chrono::system_clock;

    cv::Vec3f pos;
    cv::Vec3f vel;

    clock::time_point tp;
    double dt(clock::time_point now) const { return std::chrono::duration<double, clock::period>(now - tp).count(); }
    cv::Vec3f ps(clock::time_point now) const { return dt(now) * vel + pos; }
};

using ball_position_set = std::array<ball_position_desc, 4>;

struct shared_state {
    auto lock() const { return std::unique_lock{lock_}; }

    struct {
        cv::Vec3f pos = {}, rot = cv::Vec3f(1, 0, 0);
    } table;

    ball_position_set balls;

private:
    mutable kangsw::spinlock lock_;
};

struct shared_data : pipepp::base_shared_context {
    // options
    PIPEPP_DECLARE_OPTION_CLASS(shared_data);
    struct table {
        struct size {
            PIPEPP_DECLARE_OPTION_CATEGORY("Table.Size");
            PIPEPP_CATEGORY_OPTION(outer, cv::Vec2d(1.8, 0.98));
            PIPEPP_CATEGORY_OPTION(inner, cv::Vec2d(1.653, 0.823));
            PIPEPP_CATEGORY_OPTION(fit, cv::Vec2d(1.735, 0.915));
        };
        struct filter {
            PIPEPP_DECLARE_OPTION_CATEGORY("Table.Filter");
            PIPEPP_CATEGORY_OPTION(alpha_pos, 0.3);
            PIPEPP_CATEGORY_OPTION(alpha_rot, 0.3);
            PIPEPP_CATEGORY_OPTION(jump_threshold_distance, 0.1);
        };
        struct marker {
            PIPEPP_DECLARE_OPTION_CATEGORY("Table.Marker");

            PIPEPP_CATEGORY_OPTION(count_x, 9);
            PIPEPP_CATEGORY_OPTION(count_y, 5);
            PIPEPP_CATEGORY_OPTION(felt_width, 1.735f);
            PIPEPP_CATEGORY_OPTION(felt_height, 0.915f);
            PIPEPP_CATEGORY_OPTION(dist_from_felt_long, 0.012f);
            PIPEPP_CATEGORY_OPTION(dist_from_felt_short, 0.012f);
            PIPEPP_CATEGORY_OPTION(step, 0.206f);
            PIPEPP_CATEGORY_OPTION(width_shift_a, 0.0f);
            PIPEPP_CATEGORY_OPTION(width_shift_b, 0.0f);
            PIPEPP_CATEGORY_OPTION(height_shift_a, 0.0f);
            PIPEPP_CATEGORY_OPTION(height_shift_b, 0.01f);
        };
    };

    PIPEPP_OPTION(camera_FOV, cv::Vec2d(84.855, 53.27), "Common");

    struct ball {
        PIPEPP_OPTION(radius, 0.030239439175, "Ball");
    };

    // data
    std::shared_ptr<shared_state> state;

    recognizer_t::frame_desc imdesc_bkup;
    recognizer_t::process_finish_callback_type callback;

    cv::Mat debug_mat;

    cv::Mat cluster_color_mat;
    cv::Mat rgb, hsv;
    cv::UMat u_rgb, u_hsv;

    struct {
        std::vector<cv::Vec2f> contour;

        cv::Vec3f pos, rot;
        float confidence;
    } table;

    static void get_marker_points_model(pipepp::detail::option_base const& ec, std::vector<cv::Vec3f>& model);
};

struct input_resize {
    PIPEPP_DECLARE_OPTION_CLASS(input_resize);
    PIPEPP_OPTION(desired_image_width, 1280);

    PIPEPP_OPTION(show_sources, false, "Debug");
    PIPEPP_OPTION(test_color_spaces, false, "Debug");

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
    PIPEPP_OPTION(table_color_filter_0_lo, cv::Vec3b(175, 150, 60), "Filter");
    PIPEPP_OPTION(table_color_filter_1_hi, cv::Vec3b(10, 255, 255), "Filter");

    PIPEPP_OPTION(show_debug_mat, false, "Debug");

    PIPEPP_OPTION(area_threshold_ratio, 0.1, "Contour", "Minimul pixel size of table candidate area");
    PIPEPP_OPTION(approx_epsilon_preprocess, 5.0, "Contour", "Epsilon value used for approximate table contours");
    PIPEPP_OPTION(approx_epsilon_convex_hull, 1.0, "Contour", "Epsilon value used for approximate table contours after convex hull operation.");

    struct preprocess {
        PIPEPP_OPTION(num_erode_prev, 2, "Preprocess", "Number of erode operation before apply dilate. Dilate count determined automatically. ");
        PIPEPP_OPTION(num_erode_post, 6, "Preprocess", "Number of dilate operation after apply dilate. Dilate count is determined automatically.");
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

struct output_pipe {
    PIPEPP_DECLARE_OPTION_CLASS(output_pipe);

    struct debug {
        PIPEPP_DECLARE_OPTION_CATEGORY("Debug");
        PIPEPP_CATEGORY_OPTION(show_debug_mat, false);
    };

    using input_type = shared_data*;
    using output_type = nullptr_t;

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data& sd, input_type& i) { i = &sd; }
    static auto factory() { return pipepp::make_executor<output_pipe>(); }
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

    PIPEPP_OPTION(debug_show_partial_glyphs, true, "Debug");

    PIPEPP_OPTION(enable_partial_solver, true, "Flag");
    PIPEPP_OPTION(enable_partial_parallel_solve, true, "Flag");

    struct partial {
        PIPEPP_DECLARE_OPTION_CATEGORY("Partial");
        struct solver {
            PIPEPP_DECLARE_OPTION_CATEGORY("Partial.Solver");
            PIPEPP_CATEGORY_OPTION(iteration, 5);
            PIPEPP_CATEGORY_OPTION(candidates, 266);
            PIPEPP_CATEGORY_OPTION(rotation_axis_variant, 0.008f);
            PIPEPP_CATEGORY_OPTION(rotation_amount_variant, 0.007f);
            PIPEPP_CATEGORY_OPTION(distance_variant, 0.12f);
            PIPEPP_CATEGORY_OPTION(border_margin, 3);
            PIPEPP_CATEGORY_OPTION(iteration_narrow_rate, 0.5f);
            PIPEPP_CATEGORY_OPTION(error_function_base, 1.06);
        };

        PIPEPP_OPTION(cull_window_top_left, cv::Vec2d(0.05, 0.05), "Partial.Cull");
        PIPEPP_OPTION(cull_window_bottom_right, cv::Vec2d(0.95, 0.95), "Partial.Cull");

        PIPEPP_OPTION(apply_weight, 0.3, "Partial");
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

struct marker_finder {
    PIPEPP_DECLARE_OPTION_CLASS(marker_finder);
    PIPEPP_OPTION(enable_debug_glyphs, true, "debug");
    PIPEPP_OPTION(enable_debug_mats, true, "debug");
    PIPEPP_OPTION(show_marker_area_mask, false, "debug");

    PIPEPP_OPTION(num_insert_contour_vtx, 5, "PP 0: Marker area");
    PIPEPP_OPTION(table_border_range_outer, 0.06, "PP 0: Marker area");
    PIPEPP_OPTION(table_border_range_inner, 0.02, "PP 0: Marker area");

    PIPEPP_OPTION(laplacian_mask_threshold, 0.5, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_min_rad, 0.5, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_max_rad, 10.0, "PP 1: filtering");
    PIPEPP_OPTION(marker_area_min_size, 1, "PP 1: filtering");

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
        std::vector<cv::Vec2f> markers;
        std::vector<float> weights;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const& sd, table_edge_solver::output_type const& i, input_type& o);
};

struct marker_solver {
    PIPEPP_DECLARE_OPTION_CLASS(marker_solver);
    PIPEPP_OPTION(enable_debug_glyphs, true, "debug");
    PIPEPP_OPTION(enable_debug_mats, true, "debug");

    struct solver {
        PIPEPP_DECLARE_OPTION_CATEGORY("Solver");

        PIPEPP_CATEGORY_OPTION(iteration, 5);
        PIPEPP_CATEGORY_OPTION(error_base, 1.14);
        PIPEPP_CATEGORY_OPTION(variant_rot, 0.1);
        PIPEPP_CATEGORY_OPTION(variant_pos, 0.1);
        PIPEPP_CATEGORY_OPTION(variant_rot_axis, 0.005);
        PIPEPP_CATEGORY_OPTION(narrow_rate_pos, 0.5);
        PIPEPP_CATEGORY_OPTION(narrow_rate_rot, 0.5);
        PIPEPP_CATEGORY_OPTION(num_cands, 600);
        PIPEPP_CATEGORY_OPTION(num_iter, 5);
        PIPEPP_CATEGORY_OPTION(do_parallel, true);
        PIPEPP_CATEGORY_OPTION(confidence_amp, 1.5);
        PIPEPP_CATEGORY_OPTION(min_valid_marker_size, 1.2);
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

        std::vector<cv::Vec3f> marker_model;
        std::vector<cv::Vec2f> markers;
        std::vector<float> weights;
    };

    struct output_type {
        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
        float confidence;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const& sd, marker_finder::output_type const& i, input_type& o);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct ball_search {
    PIPEPP_DECLARE_OPTION_CLASS(ball_search);
    PIPEPP_OPTION(show_debug_mat, false, "Debug");
    PIPEPP_OPTION(show_random_sample, false, "Debug");
    PIPEPP_OPTION(random_sample_scale, 200, "Debug");
    PIPEPP_OPTION(top_view_scale, 600, "Debug", "Pixels per meter");

    struct field {
        inline static const std::string _category = "Balls.";

        struct red {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "Red");
            PIPEPP_CATEGORY_OPTION(color, cv::Vec2f(130, 205), "Representative Hue, Saturation color");
            PIPEPP_CATEGORY_OPTION(weight_hs, cv::Vec2f(4, 1), "Weight per channel");
            PIPEPP_CATEGORY_OPTION(error_fn_base, 1300000.0, "Weight per channel");
            PIPEPP_CATEGORY_OPTION(suitability_threshold, 0.2);
            PIPEPP_CATEGORY_OPTION(matching_negative_weight, 1.2);
            PIPEPP_CATEGORY_OPTION(confidence_threshold, 0.5);

            PIPEPP_CATEGORY_OPTION(second_ball_erase_radius_adder, 5);
        };

        struct orange {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "Orange");
            PIPEPP_CATEGORY_OPTION(color, cv::Vec2f(90, 210), "Representative Hue, Saturation color");
            PIPEPP_CATEGORY_OPTION(weight_hs, cv::Vec2f(2, 3), "Weight per channel");
            PIPEPP_CATEGORY_OPTION(error_fn_base, 30000.0, "Weight per channel");
            PIPEPP_CATEGORY_OPTION(suitability_threshold, 0.35);
            PIPEPP_CATEGORY_OPTION(matching_negative_weight, 3.0);
            PIPEPP_CATEGORY_OPTION(confidence_threshold, 0.5);
        };

        struct white {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "White");
            PIPEPP_CATEGORY_OPTION(color, cv::Vec2f(80, 40), "Representative Hue, Saturation color");
            PIPEPP_CATEGORY_OPTION(weight_hs, cv::Vec2f(0.2f, 1.0f), "Weight per channel");
            PIPEPP_CATEGORY_OPTION(error_fn_base, 30000.0, "Weight per channel");
            PIPEPP_CATEGORY_OPTION(suitability_threshold, 0.22);
            PIPEPP_CATEGORY_OPTION(matching_negative_weight, 3.0);
            PIPEPP_CATEGORY_OPTION(confidence_threshold, 0.5);
        };
    };

    struct random_sample {
        static bool verify_positive_area(cv::Vec2f& ref)
        {
            if (ref[0] > 1 || ref[1] > 1 || ref[0] < 0 || ref[1] < 0) {
                ref[0] = std::clamp(ref[0], 0.f, 1.f);
                ref[1] = std::clamp(ref[1], 0.f, 1.f);
                return false;
            }
            return true;
        }

        PIPEPP_DECLARE_OPTION_CATEGORY("Random Sample");
        PIPEPP_CATEGORY_OPTION(positive_area, cv::Vec2f(0.3, 1), "", &verify_positive_area);
        PIPEPP_CATEGORY_OPTION(negative_area, cv::Vec2f(1.03, 1.33));
        PIPEPP_CATEGORY_OPTION(random_seed, 42);
        PIPEPP_CATEGORY_OPTION(integral_radius, 25);
        PIPEPP_CATEGORY_OPTION(rotate_angle, 0);
    };

    struct matching {
        PIPEPP_DECLARE_OPTION_CATEGORY("Matching");
        PIPEPP_CATEGORY_OPTION(confidence_weight, 1.0);
        PIPEPP_CATEGORY_OPTION(cushion_center_gap, 0.0);
        PIPEPP_CATEGORY_OPTION(min_pixel_radius, 1);
        PIPEPP_CATEGORY_OPTION(num_candidate_dilate, 6);
        PIPEPP_CATEGORY_OPTION(num_candidate_erode, 6);
        PIPEPP_CATEGORY_OPTION(num_maximum_sample, 100000);
        PIPEPP_CATEGORY_OPTION(enable_parallel, true);
    };

    struct movement {
        PIPEPP_DECLARE_OPTION_CATEGORY("Movement");
        PIPEPP_CATEGORY_OPTION(max_error_speed, 1.3);
        PIPEPP_CATEGORY_OPTION(alpha_position, 1);
        PIPEPP_CATEGORY_OPTION(jump_distance, 0.013);
    };

    struct input_type {
        pipepp::detail::option_base const* opt_shared;
        imgproc::img_t const* imdesc;
        cv::Mat const* debug_mat;

        ball_position_set prev_ball_pos;

        cv::UMat u_rgb, u_hsv;

        cv::Size img_size;
        std::vector<cv::Vec2f> const* table_contour;

        cv::Vec3f table_pos;
        cv::Vec3f table_rot;

        cv::Vec2f table_inner_size;
        cv::Vec2f table_fit_size;
        cv::Vec2f table_outer_size;
    };
    struct output_type {
        ball_position_set new_set;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& input, output_type& out);
    static void link_from_previous(shared_data const& sd, marker_solver::output_type const& i, input_type& o);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);

private:
    std::vector<cv::Vec2f> normal_random_samples_;
    std::vector<cv::Vec2f> normal_negative_samples_;
};

} // namespace billiards::pipes