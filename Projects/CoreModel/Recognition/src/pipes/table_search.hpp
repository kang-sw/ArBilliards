#pragma once
#include "recognizer.hpp"

namespace billiards::pipes
{
class clustering
{
public:
    PIPEPP_DECLARE_OPTION_CLASS(clustering);
    PIPEPP_OPTION(target_image_width, 1280, "Common");
    PIPEPP_OPTION(true_SLIC_false_SEEDS, true, "flags");

    struct SEEDS {
        PIPEPP_OPTION(num_iter, 4, "SEEDS");
    };
    PIPEPP_OPTION(num_segments, 1024, "SEEDS");
    PIPEPP_OPTION(num_levels, 5, "SEEDS");

    PIPEPP_OPTION(show_segmentation_result, true, "debug");
    PIPEPP_OPTION(segmentation_devider_color, cv::Vec3b(255, 0, 255), "debug");

    struct SLIC {
        PIPEPP_OPTION(num_iter, 4, "SLIC");
    };
    PIPEPP_OPTION(algo_index_SLICO_MSLIC_SLIC, 0, "SLIC");
    PIPEPP_OPTION(region_size, 20, "SLIC");
    PIPEPP_OPTION(ruler, 20, "SLIC");

    struct input_type {
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