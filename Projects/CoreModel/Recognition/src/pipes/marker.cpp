#include "marker.hpp"
#include <amp.h>
#include <amp_math.h>
#include <random>

#undef max
#undef min

struct billiards::pipes::table_marker_finder::impl {
    // array<>
    size_t positive_index_fence = 0;
    std::optional<concurrency::array<cv::Vec3f>> kernel_vertexes;
};

pipepp::pipe_error billiards::pipes::table_marker_finder::operator()(pipepp::execution_context& ec, input_type const& in, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto& m = *impl_;
    bool const option_dirty = ec.consume_option_dirty_flag();

    if (option_dirty) {
        PIPEPP_ELAPSE_SCOPE("Regenerate Kernels");

        // 컨투어 랜덤 샘플 피봇 재생성
        auto _positive = kernel::positive_area(ec);
        auto _negative = kernel::negative_area(ec);
        auto positive_integral_radius = kernel::generator_positive_radius(ec);
        auto negative_integral_radius = kernel::generator_negative_radius(ec);
        std::mt19937 rand(kernel::random_seed(ec));
        std::uniform_real_distribution<float> positive(_positive[0], _positive[1]);
        std::uniform_real_distribution<float> negative(_negative[0], _negative[1]);

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

        // GPU로 데이터 복사
        m.kernel_vertexes.emplace(vtxs.size(), vtxs.begin());

        if (debug::show_generated_kernel(ec)) {
            PIPEPP_ELAPSE_SCOPE("View kernels");
            auto scale = debug::kernel_view_scale(ec);
            auto mult = scale / 4;
            auto radius = std::max(1, scale / 100);
            cv::Mat3b kernel_view(scale, scale, {0, 0, 0});
            cv::Point center(scale / 2, scale / 2);
            cv::Scalar colors[] = {{0, 255, 0}, {0, 0, 255}};

            for (auto idx : kangsw::counter(vtxs.size())) {
                auto vtx = vtxs[idx];
                cv::Point pt(vtx[0] * mult, -vtx[2] * mult);
                cv::circle(kernel_view, center + pt, radius, colors[idx >= m.positive_index_fence]);
            }

            PIPEPP_STORE_DEBUG_DATA("Generated kernel", (cv::Mat)kernel_view);
        }
    }

    return {};
}

billiards::pipes::table_marker_finder::table_marker_finder()
    : impl_(std::make_unique<impl>())
{
}

billiards::pipes::table_marker_finder::~table_marker_finder() = default;
