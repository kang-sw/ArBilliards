#pragma once
#include "pipepp/pipe.hpp"
#include "../recognition.hpp"

namespace billiards::pipes
{
struct shared_data : pipepp::base_shared_context {
    // options
    PIPEPP_DEFINE_OPTION_CLASS(shared_data);
    PIPEPP_DEFINE_OPTION_2(table_size_outer, cv::Vec2f(1, 2));
    PIPEPP_DEFINE_OPTION_2(table_size_inner, cv::Vec2f(1, 2));
    PIPEPP_DEFINE_OPTION_2(table_size_fit, cv::Vec2f(1, 2));

    // data
    recognizer_t::parameter_type param_bkup;
    recognizer_t::process_finish_callback_type callback;

    cv::Mat rgb, hsv;
    cv::UMat u_rgb, u_hsv;
};

} // namespace billiards::pipes