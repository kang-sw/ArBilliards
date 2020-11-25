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
    std::vector<cv::Vec2f> _kernel_mem;

    struct {
        std::optional<concurrency::array<cv::Vec2f>> kernel_vertexes;
        std::optional<concurrency::array<float, 2>> distances;
    } gpu;
};

void billiards::pipes::helpers::table_edge_extender::operator()(pipepp::execution_context& ec, cv::Mat& marker_area_mask)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    auto& img = *p_imdesc;
    bool const draw_debug_glyphs = !!debug_mat;
    auto& debug = *debug_mat;

    {
        PIPEPP_ELAPSE_SCOPE("Calculate Table Contour Mask");
        vector<Vec2f> contour;

        // 테이블 평면 획득
        auto table_plane = plane_t::from_rp(table_rot, table_pos, {0, 1, 0});
        plane_to_camera(img, table_plane, table_plane);

        // contour 개수를 단순 증식합니다.
        for (int i = 0, num_insert = num_insert_contour_vertexes;
             i < table_contour.size() - 1;
             ++i) {
            auto p0 = table_contour[i];
            auto p1 = table_contour[i + 1];

            for (int idx = 0; idx < num_insert + 1; ++idx) {
                contour.push_back(p0 + (p1 - p0) * (1.f / num_insert * idx));
            }
        }

        auto outer_contour = contour;
        auto inner_contour = contour;
        auto m = moments(contour);
        auto center = Vec2f(m.m10 / m.m00, m.m01 / m.m00);
        auto mass = sqrt(m.m00);
        double frame_width_outer = table_border_range_outer;
        double frame_width_inner = table_border_range_inner;

        // 각 컨투어의 거리 값에 따라, 차등적으로 밀고 당길 거리를 지정합니다.
        for (int index = 0; index < contour.size(); ++index) {
            Vec2i pt = contour[index];
            auto& outer = outer_contour[index];
            auto& inner = inner_contour[index];

            // 거리 획득
            // auto depth = img.depth.at<float>((Point)pt);
            // auto drag_width_outer = min(300.f, get_pixel_length(img, frame_width_outer, depth));
            // auto drag_width_inner = min(300.f, get_pixel_length(img, frame_width_inner, depth));
            float drag_width_outer = get_pixel_length_on_contact(img, table_plane, pt, frame_width_outer);
            float drag_width_inner = get_pixel_length_on_contact(img, table_plane, pt, frame_width_inner);

            // 평면과 해당 방향 시야가 이루는 각도 theta를 구하고, cos(theta)를 곱해 화면상의 픽셀 드래그를 구합니다.
            Vec3f pt_dir(pt[0], pt[1], 1);
            get_point_coord_3d(img, pt_dir[0], pt_dir[1], 1);
            pt_dir = normalize(pt_dir);
            auto cos_theta = abs(pt_dir.dot(table_plane.N));
            drag_width_outer *= cos_theta;
            drag_width_inner *= cos_theta;
            drag_width_outer = isnan(drag_width_outer) ? 1 : drag_width_outer;
            drag_width_inner = isnan(drag_width_inner) ? 1 : drag_width_inner;
            drag_width_outer = clamp<float>(drag_width_outer, 1, 100);
            drag_width_inner = clamp<float>(drag_width_inner, 1, 100);

            auto direction = /*normalize*/ (outer - center);
            outer += direction * drag_width_outer / mass;
            if (!is_border_pixel({{}, marker_area_mask.size()}, inner)) {
                inner -= direction * drag_width_inner / mass;
            }
        }

        vector<Vec2i> drawer;
        drawer.assign(outer_contour.begin(), outer_contour.end());
        drawContours(marker_area_mask, vector{{drawer}}, -1, 255, -1);
        if (draw_debug_glyphs) {
            drawContours(debug, vector{{drawer}}, -1, {0, 0, 0}, 2);
        }

        drawer.assign(inner_contour.begin(), inner_contour.end());
        drawContours(marker_area_mask, vector{{drawer}}, -1, 0, -1);
        if (draw_debug_glyphs) {
            drawContours(debug, vector{{drawer}}, -1, {0, 0, 0}, 2);
        }
    }
}

pipepp::pipe_error billiards::pipes::table_marker_finder::operator()(pipepp::execution_context& ec, input_type const& in, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto& m = *impl_;
    bool const option_dirty = ec.consume_option_dirty_flag();
    auto& imdesc = *in.p_imdesc;
    bool const show_debug = show_debug_mats(ec);

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

        if (show_debug) {
            helpers::kernel_visualizer kv;
            kv.kernel_view_size = debug::kernel_view_size(ec);
            kv.positive_index_fence = m.positive_index_fence;
            kv.vtxs = vtxs;

            PIPEPP_STORE_DEBUG_DATA("Generated kernel", (cv::Mat)kv(ec));
        }

        // 반지름을 곱합니다.
        for (auto radius = marker::radius(ec);
             auto& vtx : vtxs) {
            vtx *= radius;
        }
        m.kernel_model = std::move(vtxs);
    }

    auto& kernel = m._kernel_mem;
    PIPEPP_ELAPSE_BLOCK("Project points by view")
    {
        // 1 meter 거리에 대해 모든 포인트를 투사합니다.
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

        kernel.clear();
        project_model_local(imdesc, kernel, vertices, false, {});

        if (show_debug) {
            PIPEPP_ELAPSE_SCOPE("Kernel visualization");
            auto scale = debug::kernel_view_size(ec);
            auto mult = debug::current_kernel_view_scale(ec);
            auto radius = std::max(1, scale / 100);
            cv::Mat3b kernel_view(scale, scale, {0, 0, 0});
            cv::Point center(scale / 2, scale / 2);
            cv::Scalar colors[] = {{0, 255, 0}, {0, 0, 255}};
            cv::Vec2f cam_center(imdesc.camera.cx, imdesc.camera.cy);

            for (auto idx : kangsw::counter(kernel.size())) {
                auto fpt = kernel[idx];
                fpt = (fpt - cam_center) * mult;

                cv::circle(kernel_view, center + (cv::Point)(cv::Vec2i)fpt, radius, colors[idx >= m.positive_index_fence]);
            }

            PIPEPP_STORE_DEBUG_DATA("Current kernel", (cv::Mat)kernel_view);
        }
    }

    // 커널을 유효한 부분에서만 계산할 수 있도록, 먼저 컨투어를 유효 영역으로 확장합니다.
    cv::Mat1b area_mask(in.domain.size(), 0);
    PIPEPP_ELAPSE_BLOCK("Table contour area extension")
    {
        helpers::table_edge_extender tee = {
          .table_rot = in.init_table_rot,
          .table_pos = in.init_table_pos,
          .p_imdesc = in.p_imdesc,
          .table_contour = in.contour,
          .num_insert_contour_vertexes = marker::pp::num_inserted_contours(ec),
          .table_border_range_outer = marker::pp::marker_range_outer(ec),
          .table_border_range_inner = marker::pp::marker_range_inner(ec),
          .debug_mat = (cv::Mat*)&in.debug,
        };
        tee(ec, area_mask);

        PIPEPP_STORE_DEBUG_DATA_COND("Marker Area Mask", (cv::Mat)area_mask, show_debug);
    }

    // 유효 픽셀만을 필터링합니다. 두 가지 메소드 중 하나를 적용하게 됩니다.
    // (델타 색상 또는 범위 필터)
    std::vector<cv::Vec2i> valid_marker_pixels;
    PIPEPP_ELAPSE_BLOCK("Filter valid marker pixels")
    {
        using namespace imgproc;
        cv::Mat1b valid_pxls;

        if (in.lightness.empty()) {
            // 0번 메소드 시 lightness가 지정되지 않습니다.
            // 0번 메소드 ==> Range filter 후 모든 픽셀 선택
            range_filter(in.domain, valid_pxls, marker::filter::method_0_range_lo(ec), marker::filter::method_0_range_hi(ec));
        } else {
            // 다른 메소드에서는 lightness가 지정됩니다.
            // 먼저 lightness Mat에 2D 필터 적용
            cv::Matx33f edge_kernel(-1, -1, -1, -1, 8, -1, -1, -1, -1);

        }
    }

    return {};
}

billiards::pipes::table_marker_finder::table_marker_finder()
    : impl_(std::make_unique<impl>())
{
}

billiards::pipes::table_marker_finder::~table_marker_finder() = default;
