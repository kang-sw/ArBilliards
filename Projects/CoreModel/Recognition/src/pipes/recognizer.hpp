#pragma once
#include <opencv2/core/matx.hpp>

#include "shared_data.hpp"
#include "../recognition.hpp"
#include "pipepp/pipeline.hpp"

namespace billiards::pipes
{
auto build_pipe() -> std::shared_ptr<pipepp::pipeline<shared_data, struct input_resize>>;

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
    PIPEPP_DEFINE_OPTION_2(table_color_filter_lo, cv::Vec3b(0, 0, 0));
    PIPEPP_DEFINE_OPTION_2(table_color_filter_hi, cv::Vec3b(180, 255, 255));

    using input_type = input_resize::output_type;
    struct output_type {
        std::vector<cv::Vec2f> table_contour_candidate;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
};

} // namespace billiards::pipes