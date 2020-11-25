#include "marker.hpp"
#include "../image_processing.hpp"
#include <amp.h>
#include <amp_math.h>
#include <random>

#undef max
#undef min

struct billiards::pipes::table_marker_finder::impl {
    // array<>
    size_t positive_index_fence = 0;
    std::vector<cv::Vec3f> kernel_model;

    struct {
        std::optional<concurrency::array<cv::Vec2f>> kernel_vertexes;
        std::optional<concurrency::array<float, 2>> distances;
    } gpu;
};

pipepp::pipe_error billiards::pipes::table_marker_finder::operator()(pipepp::execution_context& ec, input_type const& in, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto& m = *impl_;
    bool const option_dirty = ec.consume_option_dirty_flag();
    auto& imdesc = *in.p_imdesc;

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

        if (debug::show_generated_kernel(ec)) {
            PIPEPP_ELAPSE_SCOPE("Kernel visualization");
            auto scale = debug::kernel_view_size(ec);
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

        // 반지름을 곱합니다.
        for (auto radius = marker::radius(ec);
             auto& vtx : vtxs) {
            vtx *= radius;
        }
        m.kernel_model = std::move(vtxs);
    }

    // 1 meter 거리에 대해 모든 포인트를 투사합니다.
    PIPEPP_ELAPSE_BLOCK("Project points by view")
    {
        using namespace imgproc;
        auto vertices = m.kernel_model;
        cv::Vec3f local_loc = {}, local_rot = in.init_table_rot;
        world_to_camera(imdesc, local_rot, local_loc);

        // 항상 카메라에서 1미터 떨어진 위치에서의 크기를 계산합니다.
        local_loc = {0, 0, 1};

        for (auto rotation = rodrigues(local_rot);
             auto& vt : vertices) {
            vt = rotation * vt + local_loc;
        }

        std::vector<cv::Vec2f> projected;
        project_model_local(imdesc, projected, vertices, false, {});

        if (debug::show_current_3d_kernel(ec)) {
            PIPEPP_ELAPSE_SCOPE("Kernel visualization");
            auto scale = debug::kernel_view_size(ec);
            auto mult = debug::current_kernel_view_scale(ec);
            auto radius = std::max(1, scale / 100);
            cv::Mat3b kernel_view(scale, scale, {0, 0, 0});
            cv::Point center(scale / 2, scale / 2);
            cv::Scalar colors[] = {{0, 255, 0}, {0, 0, 255}};
            cv::Vec2f cam_center(imdesc.camera.cx, imdesc.camera.cy);

            for (auto idx : kangsw::counter(projected.size())) {
                auto fpt = projected[idx];
                fpt = (fpt - cam_center) * mult;

                cv::circle(kernel_view, center + (cv::Point)(cv::Vec2i)fpt, radius, colors[idx >= m.positive_index_fence]);
            }
            for (auto idx : kangsw::counter(projected.size())) {
                auto fpt = projected[idx];
                fpt = (fpt - cam_center) * mult;

                cv::circle(kernel_view, center + (cv::Point)(cv::Vec2i)fpt, 0, colors[idx >= m.positive_index_fence]);
            }

            PIPEPP_STORE_DEBUG_DATA("Current kernel", (cv::Mat)kernel_view);
        }
    }
    return {};
}

billiards::pipes::table_marker_finder::table_marker_finder()
    : impl_(std::make_unique<impl>())
{
}

billiards::pipes::table_marker_finder::~table_marker_finder() = default;
