#include "recognizer.hpp"

#include <random>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "fmt/format.h"

#pragma warning(disable : 4244)

namespace billiards
{
class recognizer_t;
}

auto billiards::pipes::build_pipe() -> std::shared_ptr<pipepp::pipeline<shared_data, input_resize>>
{
    auto pl = decltype(build_pipe())::element_type::create(
      "input", 1, &pipepp::make_executor<pipes::input_resize>);

    auto input = pl->front();
    input.add_output_handler(&input_resize::output_handler);

    auto contour_search = input.create_and_link_output("contour search", false, 1, &contour_candidate_search::link_from_previous, &pipepp::make_executor<contour_candidate_search>);
    contour_search.add_output_handler(&contour_candidate_search::output_handler);

    auto pnp_solver = contour_search.create_and_link_output("table edge solver", false, 1, &table_edge_solver::link_from_previous, &pipepp::make_executor<table_edge_solver>);
    pnp_solver.add_output_handler(&table_edge_solver::output_handler);

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

        PIPEPP_STORE_DEBUG_DATA_COND("Filtered Image", filtered.getMat(cv::ACCESS_FAST).clone(), debug_show_0_filtered(ec));
        PIPEPP_STORE_DEBUG_DATA_COND("Edge Image", edge.getMat(cv::ACCESS_FAST).clone(), debug_show_1_edge(ec));
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

            drawContours(debug, vector{{max_size_contour}}, -1, {0, 0, 0}, 3);
        }
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

    PIPEPP_STORE_DEBUG_DATA("Debug Mat", i.debug_mat);
    return pipepp::pipe_error::ok;
}

void billiards::pipes::table_edge_solver::link_from_previous(shared_data const& sd, contour_candidate_search::output_type const& i, input_type& o)
{
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
    float pos_alpha = sd.table_filter_alpha_pos(sd);
    float rot_alpha = sd.table_filter_alpha_rot(sd);
    float jump_thr = !o.can_jump * 1e10 + sd.table_filter_jump_threshold_distance(sd);
    state.table.pos = imgproc::set_filtered_table_pos(state.table.pos, o.table_pos, pos_alpha * o.confidence, jump_thr);
    state.table.rot = imgproc::set_filtered_table_rot(state.table.rot, o.table_rot, rot_alpha * o.confidence, jump_thr);
}
