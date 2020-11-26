// Intentionally not-include-guarded.
// This file should not be included twice!
#include "recognizer.hpp"

namespace billiards::pipes {
using namespace std::literals;

namespace helpers {
struct kernel_visualizer {
    std::span<cv::Vec3f> vtxs;
    int kernel_view_size = 200;
    size_t positive_index_fence = 0;

    cv::Mat3b operator()(pipepp::execution_context& ec);
};

struct kernel_generator {
    // inputs
    std::uniform_real_distribution<float> positive, negative;
    unsigned positive_integral_radius;
    unsigned negative_integral_radius;
    unsigned random_seed;

    bool show_debug;
    unsigned kernel_view_size;

    struct out_t {
        size_t positive_index_fence;
    } output;

    std::vector<cv::Vec3f> operator()(pipepp::execution_context& ec);
};

} // namespace helpers
} // namespace billiards::pipes