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
    PIPEPP_OPTION_AUTO(debug_show_mats, true, "Debug");

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

        cv::Mat                         debug_mat;
        recognizer_t::frame_desc const* img_ptr;

        cv::Size img_size;

        std::vector<cv::Vec2f> const* table_contour;
        cv::Vec2f                     table_fit_size;

        cv::Vec3f table_pos_init;
        cv::Vec3f table_rot_init;
    };

    struct output_type {
        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
        float     confidence;
        bool      can_jump;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void        link_from_previous(shared_data const& sd, input_type& o);
    static void        output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

} // namespace billiards::pipes