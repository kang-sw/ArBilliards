#pragma once
#include "recognizer.hpp"

namespace billiards::pipes
{
class clustering
{
public:
    PIPEPP_DEFINE_OPTION_CLASS(clustering);
    PIPEPP_DEFINE_OPTION_2(num_iter, 10, "slic");
    PIPEPP_DEFINE_OPTION_2(num_segments, 1024, "slic");
    PIPEPP_DEFINE_OPTION_2(spxl_size, 50, "slic");
    PIPEPP_DEFINE_OPTION_2(coh_weight, 50.f, "slic");
    PIPEPP_DEFINE_OPTION_2(do_enforce_connectivity, false, "slic");
    PIPEPP_DEFINE_OPTION_2(true_segments_else_size, true, "slic");

    PIPEPP_DEFINE_OPTION_2(show_segmentation_result, true, "debug");

    struct input_type {
        cv::Mat rgb;
    };

    struct output_type {
        cv::Mat1i labels;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& o);

public:
    clustering();
    ~clustering();

private:
    struct implmentation;
    std::unique_ptr<implmentation> impl_;
};

} // namespace billiards::pipes