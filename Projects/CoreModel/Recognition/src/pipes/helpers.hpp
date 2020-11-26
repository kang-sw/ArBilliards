#pragma once
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

    auto operator()(pipepp::execution_context& ec)
    {
        PIPEPP_REGISTER_CONTEXT(ec);

        // 컨투어 랜덤 샘플 피봇 재생성
        std::mt19937 rand(random_seed);
        auto& m = output;
        std::vector<cv::Vec3f> vtxs;
        vtxs.reserve(size_t((positive_integral_radius + negative_integral_radius) * CV_PI * 2));

        m.positive_index_fence = -1;
        for (auto integral_radius : {positive_integral_radius, negative_integral_radius}) {
            imgproc::circle_op(
              integral_radius,
              [&](int x, int y) {
                  auto& vtx = vtxs.emplace_back();
                  vtx[0] = x, vtx[1] = 0, vtx[2] = y;
                  vtx = cv::normalize(vtx);
              });

            if (m.positive_index_fence == -1) {
                m.positive_index_fence = vtxs.size();
            }
        }

        PIPEPP_STORE_DEBUG_DATA("Total kernel points", vtxs.size());
        PIPEPP_STORE_DEBUG_DATA("Num positive points", m.positive_index_fence);
        PIPEPP_STORE_DEBUG_DATA("Num negative points", vtxs.size() - m.positive_index_fence);

        // Positive, negative 범위 할당
        for (auto idx : kangsw::counter(vtxs.size())) {
            auto& vtx = vtxs[idx];
            vtx *= idx < m.positive_index_fence ? positive(rand) : negative(rand);
        }

        if (show_debug) {
            helpers::kernel_visualizer kv;
            kv.kernel_view_size = kernel_view_size;
            kv.positive_index_fence = m.positive_index_fence;
            kv.vtxs = vtxs;

            PIPEPP_STORE_DEBUG_DATA("Generated kernel", (cv::Mat)kv(ec));
        }

        return std::move(vtxs);
    }
};

} // namespace helpers
} // namespace billiards::pipes