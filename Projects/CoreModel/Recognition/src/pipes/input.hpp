#pragma once
#include "shared_data.hpp"
#include "../recognition.hpp"

namespace billiards::pipes
{
class input
{
public:
    PIPEPP_DEFINE_OPTION_CLASS(input);

public:
    using input_type = recognizer_t::parameter_type;
    struct output_type {
        cv::Mat resized_rgb;
        cv::Mat resized_hsv;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out) { return {}; }
};

} // namespace billiards::pipes