#pragma once
#include "recognizer.hpp"

namespace billiards::pipes
{
class clustering
{
public:
    PIPEPP_DEFINE_OPTION_CLASS(clustering);
    PIPEPP_DEFINE_OPTION_2(true_SLIC_false_SEEDS, true, "flags");

    PIPEPP_DEFINE_OPTION_2(num_iter, 4, "SEEDS");
    PIPEPP_DEFINE_OPTION_2(num_segments, 1024, "SEEDS");
    PIPEPP_DEFINE_OPTION_2(num_levels, 5, "SEEDS");

    PIPEPP_DEFINE_OPTION_2(show_segmentation_result, true, "debug");

    PIPEPP_DEFINE_OPTION_2(algo_index_SLICO_MSLIC_SLIC, 0, "SLIC");
    PIPEPP_DEFINE_OPTION_2(region_size, 20, "SLIC");
    PIPEPP_DEFINE_OPTION_2(ruler, 20, "SLIC");

    struct input_type {
        cv::Mat rgb;
        cv::Mat hsv;
        cv::Mat cielab;
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

public:
    static void link_from_previous(shared_data const&, input_resize::output_type const&, input_type&);
};

} // namespace billiards::pipes