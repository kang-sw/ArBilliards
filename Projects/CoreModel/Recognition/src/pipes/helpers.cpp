#include "helpers.hpp"

cv::Mat3b billiards::pipes::helpers::kernel_visualizer::operator()(pipepp::execution_context& ec)
{
    PIPEPP_REGISTER_CONTEXT(ec);

    PIPEPP_ELAPSE_SCOPE("Kernel visualization");
    auto scale = kernel_view_size;
    auto mult = scale / 4;
    auto radius = std::max(1, scale / 100);
    cv::Mat3b kernel_view(scale, scale, {0, 0, 0});
    cv::Point center(scale / 2, scale / 2);
    cv::Scalar colors[] = {{0, 255, 0}, {0, 0, 255}};

    for (auto idx : kangsw::counter(vtxs.size())) {
        auto vtx = vtxs[idx];
        cv::Point pt(vtx[0] * mult, -vtx[2] * mult);
        cv::circle(kernel_view, center + pt, radius, colors[idx >= positive_index_fence]);
    }

    return kernel_view;
}

std::vector<cv::Vec3f> billiards::pipes::helpers::kernel_generator::operator()(pipepp::execution_context& ec)
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