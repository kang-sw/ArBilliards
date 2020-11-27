#pragma once
#include <opencv2/core/matx.hpp>

#include "../image_processing.hpp"
#include "kangsw/spinlock.hxx"
#include "pipepp/execution_context.hpp"
#include "pipepp/pipeline.hpp"
#include "recognizer.hpp"

#pragma warning(disable : 4305)

namespace billiards::pipes {
struct table_edge_solver {
    PIPEPP_DECLARE_OPTION_CLASS(table_edge_solver);

    PIPEPP_OPTION_AUTO(pnp_error_exp_fn_base,
                       1.06,
                       "PnP",
                       "Base of exponent function which is applied to calculate confidence of full-PNP solver result");
    PIPEPP_OPTION_AUTO(pnp_conf_threshold,
                       0.3,
                       "PnP",
                       "Minimum required confidence value of full-PNP solver.");

    PIPEPP_OPTION_AUTO(debug_show_partial_glyphs, true, "Debug");

    PIPEPP_OPTION_AUTO(enable_partial_solver, true, "Flag");
    PIPEPP_OPTION_AUTO(enable_partial_parallel_solve, true, "Flag");

    struct partial {
        PIPEPP_DECLARE_OPTION_CATEGORY("Partial");
        struct solver {
            PIPEPP_DECLARE_OPTION_CATEGORY("Partial.Solver");
            PIPEPP_OPTION(iteration, 5);
            PIPEPP_OPTION(candidates, 266);
            PIPEPP_OPTION(rotation_axis_variant, 0.008f);
            PIPEPP_OPTION(rotation_amount_variant, 0.007f);
            PIPEPP_OPTION(distance_variant, 0.12f);
            PIPEPP_OPTION(border_margin, 3);
            PIPEPP_OPTION(iteration_narrow_rate, 0.5f);
            PIPEPP_OPTION(error_function_base, 1.06);
        };

        PIPEPP_OPTION_AUTO(cull_window_top_left, cv::Vec2d(0.05, 0.05), "Partial.Cull");
        PIPEPP_OPTION_AUTO(cull_window_bottom_right, cv::Vec2d(0.95, 0.95), "Partial.Cull");

        PIPEPP_OPTION_AUTO(apply_weight, 0.3, "Partial");
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
    static void link_from_previous(shared_data const& sd, input_type& o);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct DEPRECATED_marker_finder {
    PIPEPP_DECLARE_OPTION_CLASS(DEPRECATED_marker_finder);
    PIPEPP_OPTION_AUTO(enable_debug_glyphs, true, "debug");
    PIPEPP_OPTION_AUTO(enable_debug_mats, true, "debug");
    PIPEPP_OPTION_AUTO(show_marker_area_mask, false, "debug");

    PIPEPP_OPTION_AUTO(num_insert_contour_vtx, 5, "PP 0: Marker area");
    PIPEPP_OPTION_AUTO(table_border_range_outer, 0.06, "PP 0: Marker area");
    PIPEPP_OPTION_AUTO(table_border_range_inner, 0.02, "PP 0: Marker area");

    PIPEPP_OPTION_AUTO(laplacian_mask_threshold, 0.5, "PP 1: filtering");
    PIPEPP_OPTION_AUTO(marker_area_min_rad, 0.5, "PP 1: filtering");
    PIPEPP_OPTION_AUTO(marker_area_max_rad, 10.0, "PP 1: filtering");
    PIPEPP_OPTION_AUTO(marker_area_min_size, 1, "PP 1: filtering");

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

struct marker_solver_OLD {
    PIPEPP_DECLARE_OPTION_CLASS(marker_solver_OLD);
    PIPEPP_OPTION_AUTO(enable_debug_glyphs, true, "debug");
    PIPEPP_OPTION_AUTO(enable_debug_mats, true, "debug");

    struct solver {
        PIPEPP_DECLARE_OPTION_CATEGORY("Solver");

        PIPEPP_OPTION(iteration, 5);
        PIPEPP_OPTION(error_base, 1.14);
        PIPEPP_OPTION(variant_rot, 0.1);
        PIPEPP_OPTION(variant_pos, 0.1);
        PIPEPP_OPTION(variant_rot_axis, 0.005);
        PIPEPP_OPTION(narrow_rate_pos, 0.5);
        PIPEPP_OPTION(narrow_rate_rot, 0.5);
        PIPEPP_OPTION(num_cands, 600);
        PIPEPP_OPTION(num_iter, 5);
        PIPEPP_OPTION(do_parallel, true);
        PIPEPP_OPTION(confidence_amp, 1.5);
        PIPEPP_OPTION(min_valid_marker_size, 1.2);
    };

    struct input_type {
        imgproc::img_t const* img_ptr;
        cv::Size img_size;

        cv::Vec3f table_pos_init;
        cv::Vec3f table_rot_init;

        cv::Mat const* debug_mat;
        std::vector<cv::Vec2f> const* p_table_contour;

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
    static void link_from_previous(shared_data const& sd, DEPRECATED_marker_finder::output_type const& i, input_type& o);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct DEPRECATED_ball_search {
    PIPEPP_DECLARE_OPTION_CLASS(DEPRECATED_ball_search);
    PIPEPP_OPTION_AUTO(show_debug_mat, false, "Debug");
    PIPEPP_OPTION_AUTO(show_random_sample, false, "Debug");
    PIPEPP_OPTION_AUTO(random_sample_scale, 200, "Debug");
    PIPEPP_OPTION_AUTO(top_view_scale, 600, "Debug", "Pixels per meter");

    struct field {
        inline static const std::string _category = "Balls.";

        struct red {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "Red");
            PIPEPP_OPTION(color, cv::Vec2f(130, 205), "Representative Hue, Saturation color");
            PIPEPP_OPTION(weight_hs, cv::Vec2f(4, 1), "Weight per channel");
            PIPEPP_OPTION(error_fn_base, 1300000.0, "Weight per channel");
            PIPEPP_OPTION(suitability_threshold, 0.2);
            PIPEPP_OPTION(matching_negative_weight, 1.2);
            PIPEPP_OPTION(confidence_threshold, 0.5);

            PIPEPP_OPTION(second_ball_erase_radius_adder, 5);
        };

        struct orange {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "Orange");
            PIPEPP_OPTION(color, cv::Vec2f(90, 210), "Representative Hue, Saturation color");
            PIPEPP_OPTION(weight_hs, cv::Vec2f(2, 3), "Weight per channel");
            PIPEPP_OPTION(error_fn_base, 30000.0, "Weight per channel");
            PIPEPP_OPTION(suitability_threshold, 0.35);
            PIPEPP_OPTION(matching_negative_weight, 3.0);
            PIPEPP_OPTION(confidence_threshold, 0.5);
        };

        struct white {
            PIPEPP_DECLARE_OPTION_CATEGORY(_category + "White");
            PIPEPP_OPTION(color, cv::Vec2f(80, 40), "Representative Hue, Saturation color");
            PIPEPP_OPTION(weight_hs, cv::Vec2f(0.2f, 1.0f), "Weight per channel");
            PIPEPP_OPTION(error_fn_base, 30000.0, "Weight per channel");
            PIPEPP_OPTION(suitability_threshold, 0.22);
            PIPEPP_OPTION(matching_negative_weight, 3.0);
            PIPEPP_OPTION(confidence_threshold, 0.5);
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
        PIPEPP_OPTION(positive_area, cv::Vec2f(0.3, 1), "", &verify_positive_area);
        PIPEPP_OPTION(negative_area, cv::Vec2f(1.03, 1.33));
        PIPEPP_OPTION(random_seed, 42);
        PIPEPP_OPTION(integral_radius, 25);
        PIPEPP_OPTION(rotate_angle, 0);
    };

    struct matching {
        PIPEPP_DECLARE_OPTION_CATEGORY("Matching");
        PIPEPP_OPTION(confidence_weight, 1.0);
        PIPEPP_OPTION(cushion_center_gap, 0.0);
        PIPEPP_OPTION(min_pixel_radius, 1);
        PIPEPP_OPTION(num_candidate_dilate, 6);
        PIPEPP_OPTION(num_candidate_erode, 6);
        PIPEPP_OPTION(num_maximum_sample, 100000);
        PIPEPP_OPTION(enable_parallel, true);
    };

    struct movement {
        PIPEPP_DECLARE_OPTION_CATEGORY("Movement");
        PIPEPP_OPTION(max_error_speed, 1.3);
        PIPEPP_OPTION(alpha_position, 1);
        PIPEPP_OPTION(jump_distance, 0.013);
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
    static void link_from_previous(shared_data const& sd, marker_solver_OLD::output_type const& i, input_type& o);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);

private:
    std::vector<cv::Vec2f> normal_random_samples_;
    std::vector<cv::Vec2f> normal_negative_samples_;
};

} // namespace billiards::pipes