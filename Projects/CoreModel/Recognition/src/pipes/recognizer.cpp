#include "recognizer.hpp"

#include <random>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include "fmt/format.h"
#include "table_search.hpp"

#pragma warning(disable : 4244)

namespace billiards
{
class recognizer_t;
}

auto billiards::pipes::build_pipe() -> std::shared_ptr<pipepp::pipeline<shared_data, input_resize>>
{
    auto pl = decltype(build_pipe())::element_type::create(
      "input", 1, &pipepp::make_executor<pipes::input_resize>);

    auto input_proxy = pl->front();
    input_proxy.add_output_handler(&input_resize::output_handler);

    { // Optional SLIC scope
        auto superpixels = input_proxy.create_and_link_output("Superpixels", true, 1, &clustering::link_from_previous, &pipepp::make_executor<clustering>);
    }

    auto contour_search_proxy = input_proxy.create_and_link_output("contour search", false, 1, &contour_candidate_search::link_from_previous, &pipepp::make_executor<contour_candidate_search>);
    contour_search_proxy.add_output_handler(&contour_candidate_search::output_handler);

    auto pnp_solver_proxy = contour_search_proxy.create_and_link_output("table edge solver", false, 1, &table_edge_solver::link_from_previous, &pipepp::make_executor<table_edge_solver>);
    pnp_solver_proxy.add_output_handler(&table_edge_solver::output_handler);

    auto marker_solver_proxy = pnp_solver_proxy.create_and_link_output("marker solver", false, 1, &marker_solver::link_from_previous, &pipepp::make_executor<marker_solver>);

    return pl;
}

pipepp::pipe_error billiards::pipes::input_resize::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto src_size = i.rgba.size();
    auto width = std::min(src_size.width, desired_image_width(ec));
    auto height = int((int64_t)width * src_size.height / src_size.width);

    PIPEPP_ELAPSE_BLOCK("Resizing")
    {
        cv::UMat rgb;

        cv::cvtColor(i.rgba, rgb, cv::COLOR_RGBA2RGB);
        if (src_size.width != width) {
            cv::resize(rgb, out.u_rgb, {width, height});
        }
        else {
            out.u_rgb = std::move(rgb);
        }
    }

    PIPEPP_ELAPSE_BLOCK("Color convert")
    {
        out.u_rgb.copyTo(out.rgb);
        cv::cvtColor(out.u_rgb, out.u_hsv, cv::COLOR_RGB2HSV);
        out.u_hsv.copyTo(out.hsv);
    }

    PIPEPP_STORE_DEBUG_DATA_COND("Source RGB", out.rgb.clone(), debug_show_source(ec));
    PIPEPP_STORE_DEBUG_DATA_COND("Source HSV", out.hsv.clone(), debug_show_hsv(ec));

    out.img_size = cv::Size(width, height);

    return pipepp::pipe_error::ok;
}

void billiards::pipes::input_resize::output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o)
{
    sd.u_hsv = o.u_hsv;
    sd.u_rgb = o.u_rgb;
    sd.hsv = o.hsv;
    sd.rgb = o.rgb;

    o.rgb.copyTo(sd.debug_mat);
}

pipepp::pipe_error billiards::pipes::contour_candidate_search::invoke(pipepp::execution_context& ec, input_type const& i, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace std;
    cv::UMat filtered, edge, u0, u1;
    cv::Vec3b filter[] = {table_color_filter_0_lo(ec), table_color_filter_1_hi(ec)};
    vector<cv::Vec2f> table_contour;
    auto image_size = i.u_hsv.size();
    auto& debug = i.debug_display;

    PIPEPP_ELAPSE_BLOCK("Edge detection")
    {
        imgproc::filter_hsv(i.u_hsv, filtered, filter[0], filter[1]);
        cv::erode(filtered, u0, {});
        cv::subtract(filtered, u0, edge);

        PIPEPP_STORE_DEBUG_DATA_COND("Filtered Image", filtered.getMat(cv::ACCESS_FAST).clone(), show_0_filtered(ec));
        PIPEPP_STORE_DEBUG_DATA_COND("Edge Image", edge.getMat(cv::ACCESS_FAST).clone(), show_1_edge(ec));
    }

    PIPEPP_ELAPSE_BLOCK("Contour Approx & Select")
    {
        using namespace cv;
        vector<vector<Vec2i>> candidates;
        vector<Vec4i> hierarchy;
        findContours(filtered, candidates, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        // 테이블 전체가 시야에 없을 때에도 위치를 추정할 수 있도록, 가장 큰 영역을 기록해둡니다.
        auto max_size_arg = make_pair(-1, 0.0);
        auto eps0 = approx_epsilon_preprocess(ec);
        auto eps1 = approx_epsilon_convex_hull(ec);
        auto size_threshold = area_threshold_ratio(ec) * image_size.area();

        // 사각형 컨투어 찾기
        for (int idx = 0; idx < candidates.size(); ++idx) {
            auto& contour = candidates[idx];

            approxPolyDP(vector(contour), contour, eps0, true);
            auto area_size = contourArea(contour);
            if (area_size < size_threshold) {
                continue;
            }

            if (max_size_arg.second < area_size) {
                max_size_arg = {idx, area_size};
            }

            convexHull(vector(contour), contour, true);
            approxPolyDP(vector(contour), contour, eps1, true);
            putText(debug, (stringstream() << "[" << contour.size() << ", " << area_size << "]").str(), contour[0], FONT_HERSHEY_PLAIN, 1.0, {0, 255, 0});

            bool const table_found = contour.size() == 4;

            if (table_found) {
                table_contour.assign(contour.begin(), contour.end());
                break;
            }
        }

        if (table_contour.empty() && max_size_arg.first >= 0) {
            auto& max_size_contour = candidates[max_size_arg.first];
            table_contour.assign(max_size_contour.begin(), max_size_contour.end());
        }
    }

    if (!table_contour.empty()) {
        vector<cv::Vec2i> pts{table_contour.begin(), table_contour.end()};
        drawContours(debug, vector{{pts}}, -1, {0, 0, 255}, 3);
    }

    o.table_contour_candidate = move(table_contour);
    return pipepp::pipe_error::ok;
}

void billiards::pipes::contour_candidate_search::link_from_previous(shared_data const& sd, input_resize::output_type const& i, input_type& o)
{
    o.u_hsv = i.u_hsv;
    o.debug_display = sd.debug_mat;
}

void billiards::pipes::contour_candidate_search::output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o)
{
    sd.table.contour = std::move(o.table_contour_candidate);
}

pipepp::pipe_error billiards::pipes::table_edge_solver::invoke(pipepp::execution_context& ec, input_type const& i, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    auto table_contour = *i.table_contour;
    auto& img = *i.img_ptr;

    o.confidence = 0;
    vector<Vec3f> obj_pts;
    get_table_model(obj_pts, i.table_fit_size);

    bool is_any_border_point = [&]() {
        for (auto pt : table_contour) {
            if (is_border_pixel({{}, i.img_size}, pt)) return true;
        }
        return false;
    }();

    if (table_contour.size() == 4 && !is_any_border_point) {
        PIPEPP_ELAPSE_SCOPE("PNP Solver");

        Vec3f tvec, rvec;
        float max_confidence = 0;

        auto [mat_cam, mat_disto] = get_camera_matx(img);

        // 테이블의 방향을 고려하여, 그대로의 인덱스와 시프트한 인덱스 각각에 대해 PnP 알고리즘을 적용, 포즈를 계산합니다.
        for (int iter = 0; iter < 2; ++iter) {
            PIPEPP_ELAPSE_SCOPE("Solve Iteration");

            Vec3d pos, rot;

            PIPEPP_ELAPSE_BLOCK("SolvePnP Time")
            if (!solvePnP(obj_pts, table_contour, mat_cam, mat_disto, rot, pos)) {
                continue;
            }

            // confidence 계산
            auto vertexes = obj_pts;
            for (auto& vtx : vertexes) {
                vtx = rodrigues(rot) * vtx + pos;
            }

            PIPEPP_ELAPSE_SCOPE("Projection");
            vector<vector<Vec2i>> contours;
            vector<Vec2f> mapped;
            project_model_local(img, mapped, vertexes, false, {});
            contours.emplace_back().assign(mapped.begin(), mapped.end());

            // 각 점을 비교하여 에러를 계산합니다.
            auto& proj = contours.front();
            double error_sum = 0;
            for (size_t index = 0; index < 4; index++) {
                Vec2f projpt = proj[index];
                error_sum += norm(projpt - table_contour[index], NORM_L2SQR);
            }

            auto conf = pow(pnp_error_exp_fn_base(ec), -sqrt(error_sum));
            if (conf > max_confidence) {
                max_confidence = conf, tvec = pos, rvec = rot;
            }

            // 점 배열 1개 회전
            table_contour.push_back(table_contour.front());
            table_contour.erase(table_contour.begin());
        }

        if (max_confidence > pnp_conf_threshold(ec)) {
            camera_to_world(img, rvec, tvec);

            o.confidence = max_confidence;
            o.table_pos = tvec;
            o.table_rot = rvec;
            o.can_jump = true;

            PIPEPP_STORE_DEBUG_DATA("Full PNP data confidence", o.confidence);
        }
    }

    if (table_contour.empty() == false && o.confidence == 0) {
        // full-point PnP 알고리즘이 실패한 경우, partial view를 수행합니다.

        vector<Vec3f> model;
        get_table_model(model, i.table_fit_size);

        auto init_pos = i.table_pos_init;
        auto init_rot = i.table_rot_init;

        int num_iteration = partial_solver_iteration(ec);
        int num_candidates = partial_solver_candidates(ec);
        float rot_axis_variant = rotation_axis_variant(ec);
        float rot_variant = rotation_amount_variant(ec);
        float pos_initial_distance = distance_variant(ec);

        vector<Vec2f> input = table_contour;
        transform_estimation_param_t param = {num_iteration, num_candidates, rot_axis_variant, rot_variant, pos_initial_distance, border_margin(ec)};
        Vec2f FOV = i.FOV_degree;
        param.FOV = {FOV[0], FOV[1]};
        param.debug_render_mat = i.debug_mat;
        param.render_debug_glyphs = true;
        param.do_parallel = enable_partial_parallel_solve(ec);
        param.iterative_narrow_ratio = iteration_narrow_rate(ec);
        param.confidence_calc_base = error_function_base(ec);

        // contour 컬링 사각형을 계산합니다.
        {
            Vec2d tl = cull_window_top_left(ec);
            Vec2d br = cull_window_bottom_right(ec);
            Vec2i img_size = static_cast<Point>(i.img_size);

            Rect r{(Point)(Vec2i)tl.mul(img_size), (Point)(Vec2i)br.mul(img_size)};
            if (get_safe_ROI_rect(i.debug_mat, r)) {
                param.contour_cull_rect = r;
            }
            else {
                param.contour_cull_rect = Rect{{}, img_size};
            }
        }

        PIPEPP_ELAPSE_SCOPE("Partial iterative search");
        auto result = estimate_matching_transform(img, input, model, init_pos, init_rot, param);

        if (result.has_value()) {
            auto& res = *result;
            float partial_weight = apply_weight(ec);
            o.confidence = res.confidence * partial_weight;
            o.table_pos = res.position;
            o.table_rot = res.rotation;

            o.can_jump = false;
            PIPEPP_STORE_DEBUG_DATA("Partial data confidence", o.confidence);
        }
    }

    if (o.confidence > 0.1) {
        using namespace cv;
        PIPEPP_ELAPSE_SCOPE("Visualize");
        auto& rvec = o.table_rot;
        auto& tvec = o.table_pos;
        draw_axes(img, i.debug_mat, rvec, tvec, 0.05f, 3);
        Scalar color = o.can_jump ? Scalar{0, 255, 0} : Scalar{0, 255, 255};
        project_contours(img, i.debug_mat, obj_pts, tvec, rvec, color, 2, {86, 58});
    }

    return pipepp::pipe_error::ok;
}

void billiards::pipes::table_edge_solver::link_from_previous(shared_data const& sd, contour_candidate_search::output_type const& i, input_type& o)
{
    auto _lck = sd.state->lock();
    o = input_type{
      .FOV_degree = sd.camera_FOV(sd),
      .debug_mat = sd.debug_mat,
      .img_ptr = &sd.param_bkup,
      .img_size = sd.debug_mat.size(),
      .table_contour = &sd.table.contour,
      .table_fit_size = sd.table_size_fit(sd),
      .table_pos_init = sd.state->table.pos,
      .table_rot_init = sd.state->table.rot};
}

void billiards::pipes::table_edge_solver::output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o)
{
    auto& state = *sd.state;
    auto _lck = state.lock();
    float pos_alpha = sd.table_filter_alpha_pos(sd);
    float rot_alpha = sd.table_filter_alpha_rot(sd);
    float jump_thr = !o.can_jump * 1e10 + sd.table_filter_jump_threshold_distance(sd);
    state.table.pos = imgproc::set_filtered_table_pos(state.table.pos, o.table_pos, pos_alpha * o.confidence, jump_thr);
    state.table.rot = imgproc::set_filtered_table_rot(state.table.rot, o.table_rot, rot_alpha * o.confidence, jump_thr);
}

pipepp::pipe_error billiards::pipes::marker_solver::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    PIPEPP_STORE_DEBUG_DATA("Debug Mat", *i.debug_mat);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    auto& img = *i.img_ptr;
    auto& debug = *i.debug_mat;
    auto table_rot = i.table_rot_init;
    auto table_pos = i.table_pos_init;
    auto& table_contour = *i.table_contour;

    out.confidence = 0;

    if (table_contour.empty()) {
        return pipepp::pipe_error::ok;
    }

    Mat1b marker_area_mask(debug.size(), 0);
    {
        PIPEPP_ELAPSE_SCOPE("Calculate Table Contour Mask");
        vector<Vec2f> contour;

        // 테이블 평면 획득
        auto table_plane = plane_t::from_rp(table_rot, table_pos, {0, 1, 0});
        plane_to_camera(img, table_plane, table_plane);

        // contour 개수를 단순 증식합니다.
        for (int i = 0, num_insert = num_insert_contour_vtx(ec);
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
        double frame_width_outer = table_border_range_outer(ec);
        double frame_width_inner = table_border_range_inner(ec);

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

            auto direction = normalize(outer - center);
            outer += direction * drag_width_outer;
            if (!is_border_pixel({{}, marker_area_mask.size()}, inner)) {
                inner -= direction * drag_width_inner;
            }
        }

        vector<Vec2i> drawer;
        drawer.assign(outer_contour.begin(), outer_contour.end());
        drawContours(marker_area_mask, vector{{drawer}}, -1, 255, -1);
        if (enable_debug_glyphs(ec)) {
            drawContours(debug, vector{{drawer}}, -1, {0, 0, 0}, 2);
        }

        drawer.assign(inner_contour.begin(), inner_contour.end());
        drawContours(marker_area_mask, vector{{drawer}}, -1, 0, -1);
        if (enable_debug_glyphs(ec)) {
            drawContours(debug, vector{{drawer}}, -1, {0, 0, 0}, 2);
        }
    }

    PIPEPP_CAPTURE_DEBUG_DATA_COND(marker_area_mask, show_marker_area_mask(ec));
    vector<Vec2f> centers;
    vector<float> marker_weights;

    PIPEPP_ELAPSE_BLOCK("Filter Marker Range")
    {
        auto& u_hsv = *i.u_hsv;
        vector<UMat> channels;
        split(u_hsv, channels);

        if (channels.size() >= 3) {
            auto& u_value = channels[2];
            UMat u0, u1;
            UMat laplacian_mask(u_hsv.size(), CV_8U, {0});
            vector<vector<Vec2i>> contours;

            // 샤프닝 적용 후 라플랑시안 적용
            PIPEPP_ELAPSE_BLOCK("Sequential filtering method")
            {
                Mat1f kernel(3, 3, -1.0 / 255.f);
                kernel(1, 1) = 8 / 255.f;
                filter2D(u_value, u1, CV_32F, kernel.getUMat(ACCESS_READ));
                PIPEPP_STORE_DEBUG_DATA_COND("Marker Filter - 1 Sharpen", u1.getMat(ACCESS_FAST).clone(), enable_debug_mats(ec));

                compare(u1, Scalar(laplacian_mask_threshold(ec)), u0, CMP_GT);
                dilate(u0, u1, {});
                erode(u1, u0, {}, {-1, -1}, 1);
                u0.copyTo(laplacian_mask, marker_area_mask);

                PIPEPP_STORE_DEBUG_DATA_COND("Marker Filter - 2 Laplacian Thresholded", laplacian_mask.getMat(ACCESS_FAST).clone(), enable_debug_mats(ec));

                findContours(laplacian_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, {});
            }

            PIPEPP_ELAPSE_SCOPE("Select valid centers");
            float rad_min = marker_area_min_rad(ec);
            float rad_max = marker_area_max_rad(ec);
            float area_min = marker_area_min_size(ec);
            for (auto& ctr : contours) {
                Point2f center;
                float radius;
                minEnclosingCircle(ctr, center, radius);
                float area = contourArea(ctr);

                if (area >= area_min && rad_min <= radius && radius <= rad_max) {
                    centers.push_back(center);
                    marker_weights.push_back(contourArea(ctr));
                }
            }
        }
    }

    return pipepp::pipe_error::ok;
}

void billiards::pipes::marker_solver::link_from_previous(shared_data const& sd, table_edge_solver::output_type const& i, input_type& o)
{
    auto _lck = sd.state->lock();
    o = input_type{
      .img_ptr = &sd.param_bkup,
      .img_size = sd.rgb.size(),
      .table_pos_init = sd.state->table.pos,
      .table_rot_init = sd.state->table.rot,
      .debug_mat = &sd.debug_mat,
      .table_contour = &sd.table.contour,
      .u_hsv = &sd.u_hsv,
    };
}
