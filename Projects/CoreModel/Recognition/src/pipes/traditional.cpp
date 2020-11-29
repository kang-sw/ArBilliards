#include "traditional.hpp"
#include "fmt/format.h"

void build_traditional_path(pipepp::pipeline<billiards::pipes::shared_data, billiards::pipes::input_resize>::initial_proxy_type input_proxy, pipepp::pipe_proxy<billiards::pipes::shared_data, billiards::pipes::output_pipe> output_pipe_proxy)
{
    // ---------------------------------------------------------------------------------
    //      CASE B: TRADITIONAL CONTOUR SEARCH
    auto contour_search_proxy
      = input_proxy.create_and_link_output(
        "contour search",
        1,
        &billiards::pipes::contour_candidate_search::link_from_previous,
        &pipepp::make_executor<billiards::pipes::contour_candidate_search>);
    contour_search_proxy.add_output_handler(&billiards::pipes::contour_candidate_search::output_handler);

    // ---------------------------------------------------------------------------------
    //  PREPROCESSOR --> [TABLE CONTOUR SOLVER]
    // ---------------------------------------------------------------------------------
    auto pnp_solver_proxy
      = contour_search_proxy.create_and_link_output(
        "table edge solver",
        2,
        &billiards::pipes::table_edge_solver::link_from_previous,
        &pipepp::make_executor<billiards::pipes::table_edge_solver>);
    pnp_solver_proxy.add_output_handler(&billiards::pipes::table_edge_solver::output_handler);
    pnp_solver_proxy.configure_tweaks().selective_input  = true;
    pnp_solver_proxy.configure_tweaks().selective_output = true;

    // ---------------------------------------------------------------------------------
    //  TABLE SOLVER --> [MARKER SEARCH]
    // ---------------------------------------------------------------------------------
    //      CASE A: TRADITIONAL METHOD
    auto marker_finder_proxy
      = pnp_solver_proxy.create_and_link_output(
        "marker finder",
        1,
        &billiards::pipes::DEPRECATED_marker_finder::link_from_previous,
        &pipepp::make_executor<billiards::pipes::DEPRECATED_marker_finder>);

    // ---------------------------------------------------------------------------------
    //      CASE B: SUPERPIXELS
    // TODO

    // ---------------------------------------------------------------------------------
    //  MARKER SEARCH --> [MARKER SOLVER]
    // ---------------------------------------------------------------------------------
    auto marker_solver_proxy
      = marker_finder_proxy.create_and_link_output(
        "marker solver",
        1,
        &billiards::pipes::marker_solver_OLD::link_from_previous,
        &pipepp::make_executor<billiards::pipes::marker_solver_OLD>);
    marker_solver_proxy.add_output_handler(&billiards::pipes::marker_solver_OLD::output_handler);

    // ---------------------------------------------------------------------------------
    //  [TABLE POSITION] --> BALL SEARCH
    // ---------------------------------------------------------------------------------
    //      CASE A: TRADITIONAL METHOD
    auto ball_finder_proxy
      = marker_solver_proxy.create_and_link_output(
        "ball finder",
        4,
        &billiards::pipes::DEPRECATED_ball_search::link_from_previous,
        &pipepp::make_executor<billiards::pipes::DEPRECATED_ball_search>);

    // ---------------------------------------------------------------------------------
    //      CASE B: SUPERPIXELS

    // ---------------------------------------------------------------------------------
    //  FINAL OUTPUT PIPE
    ball_finder_proxy.link_output(output_pipe_proxy, &billiards::pipes::output_pipe::link_from_previous);
}

pipepp::pipe_error billiards::pipes::contour_candidate_search::invoke(pipepp::execution_context& ec, input_type const& i, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace std;
    cv::UMat          u_filtered, u_edge, u0, u1;
    cv::Vec3b         filter[] = {table_color_filter_0_lo(ec), table_color_filter_1_hi(ec)};
    vector<cv::Vec2f> table_contour;
    auto              image_size = i.u_hsv.size();
    auto&             debug      = i.debug_display;

    PIPEPP_ELAPSE_BLOCK("Edge detection")
    {
        PIPEPP_ELAPSE_BLOCK("Preprocess: hsv filtering")
        {
            imgproc::range_filter(i.u_hsv, u_filtered, filter[0], filter[1]);
            imgproc::carve_outermost_pixels(u_filtered, {0});
        }

        auto prev_iter  = max(0, preprocess::num_erode_prev(ec));
        auto post_iter  = max(0, preprocess::num_erode_post(ec));
        auto num_dilate = prev_iter + post_iter;

        if (num_dilate > 0) {
            using namespace cv;

            PIPEPP_ELAPSE_SCOPE("Preprocess: erode-dilate-erode operation")
            copyMakeBorder(u_filtered, u0, num_dilate, num_dilate, num_dilate, num_dilate, BORDER_CONSTANT);
            prev_iter ? erode(u0, u1, {}, {-1, -1}, prev_iter, BORDER_CONSTANT, {}) : (void)(u1 = u0);
            dilate(u1, u0, {}, {-1, -1}, num_dilate, BORDER_CONSTANT, {});
            post_iter ? erode(u0, u1, {}, {-1, -1}, post_iter, BORDER_CONSTANT, {}) : (void)(u1 = u0);
            u_filtered = u1(Rect{{num_dilate, num_dilate}, u_filtered.size()});
        }

        cv::erode(u_filtered, u0, {});
        cv::subtract(u_filtered, u0, u_edge);

        bool const dbg_show = show_debug_mat(ec);
        PIPEPP_STORE_DEBUG_DATA_COND("Filtered Image", u_filtered.getMat(cv::ACCESS_FAST).clone(), dbg_show);
        PIPEPP_STORE_DEBUG_DATA_COND("Edge Image", u_edge.getMat(cv::ACCESS_FAST).clone(), dbg_show);
    }

    PIPEPP_ELAPSE_BLOCK("Contour Approx & Select")
    {
        using namespace cv;
        vector<vector<Vec2i>> candidates;
        vector<Vec4i>         hierarchy;
        findContours(u_filtered, candidates, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        // 테이블 전체가 시야에 없을 때에도 위치를 추정할 수 있도록, 가장 큰 영역을 기록해둡니다.
        auto max_size_arg   = make_pair(-1, 0.0);
        auto eps0           = approx_epsilon_preprocess(ec);
        auto eps1           = approx_epsilon_convex_hull(ec);
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
            PIPEPP_CAPTURE_DEBUG_DATA(max_size_arg.second);
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
    o.u_hsv         = sd.u_hsv;
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

    auto  table_contour = *i.table_contour;
    auto& img           = *i.img_ptr;

    o.confidence = 0;
    o.can_jump   = false;
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
            vector<Vec2f>         mapped;
            project_model_local(img, mapped, vertexes, false, {});
            contours.emplace_back().assign(mapped.begin(), mapped.end());

            // 각 점을 비교하여 에러를 계산합니다.
            auto&  proj      = contours.front();
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
            o.table_pos  = tvec;
            o.table_rot  = rvec;
            o.can_jump   = true;

            PIPEPP_STORE_DEBUG_DATA("Full PNP data confidence", o.confidence);
        }
    }

    if (enable_partial_solver(ec) && table_contour.empty() == false && o.confidence == 0) {
        // full-point PnP 알고리즘이 실패한 경우, partial view를 수행합니다.

        vector<Vec3f> model;
        get_table_model(model, i.table_fit_size);

        auto init_pos = i.table_pos_init;
        auto init_rot = i.table_rot_init;

        int   num_iteration        = partial::solver::iteration(ec);
        int   num_candidates       = partial::solver::candidates(ec);
        float rot_axis_variant     = partial::solver::rotation_axis_variant(ec);
        float rot_variant          = partial::solver::rotation_amount_variant(ec);
        float pos_initial_distance = partial::solver::distance_variant(ec);

        vector<Vec2f>                input = table_contour;
        transform_estimation_param_t param = {num_iteration, num_candidates, rot_axis_variant, rot_variant, pos_initial_distance, partial::solver::border_margin(ec)};
        Vec2f                        FOV   = i.FOV_degree;
        param.FOV                          = {FOV[0], FOV[1]};
        param.debug_render_mat             = i.debug_mat;
        param.render_debug_glyphs          = true;
        param.do_parallel                  = enable_partial_parallel_solve(ec);
        param.iterative_narrow_ratio       = partial::solver::iteration_narrow_rate(ec);
        param.confidence_calc_base         = partial::solver::error_function_base(ec);

        // contour 컬링 사각형을 계산합니다.
        {
            Vec2d tl       = partial::cull_window_top_left(ec);
            Vec2d br       = partial::cull_window_bottom_right(ec);
            Vec2i img_size = static_cast<Point>(i.img_size);

            Rect r{(Point)(Vec2i)tl.mul(img_size), (Point)(Vec2i)br.mul(img_size)};
            if (get_safe_ROI_rect(i.debug_mat, r)) {
                param.contour_cull_rect = r;
            } else {
                param.contour_cull_rect = Rect{{}, img_size};
            }
        }

        PIPEPP_ELAPSE_SCOPE("Partial iterative search");
        auto result = estimate_matching_transform(img, input, model, init_pos, init_rot, param);

        if (result.has_value()) {
            auto& res    = *result;
            o.confidence = res.confidence * partial::apply_weight(ec);
            o.table_pos  = res.position;
            o.table_rot  = res.rotation;

            o.can_jump = false;
            PIPEPP_STORE_DEBUG_DATA("Partial data confidence", res.confidence);
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

    PIPEPP_STORE_DEBUG_DATA_COND("Result", i.debug_mat.clone(), debug_show_mats(ec));

    return pipepp::pipe_error::ok;
}

void billiards::pipes::table_edge_solver::link_from_previous(shared_data const& sd, input_type& o)
{
    o = input_type{
      .FOV_degree     = sd.camera_FOV(sd),
      .debug_mat      = sd.debug_mat,
      .img_ptr        = &sd.imdesc_bkup,
      .img_size       = sd.debug_mat.size(),
      .table_contour  = &sd.table.contour,
      .table_fit_size = shared_data::table::size::fit(sd),
      .table_pos_init = sd.table.pos,
      .table_rot_init = sd.table.rot};
}

void billiards::pipes::table_edge_solver::output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o)
{
    float pos_alpha = shared_data::table::filter::alpha_pos(sd);
    float rot_alpha = shared_data::table::filter::alpha_rot(sd);
    float jump_thr  = !o.can_jump * 1e10 + shared_data::table::filter::jump_threshold_distance(sd);
    sd.table.pos    = imgproc::set_filtered_table_pos(sd.table.pos, o.table_pos, pos_alpha * o.confidence, jump_thr);
    sd.table.rot    = imgproc::set_filtered_table_rot(sd.table.rot, o.table_rot, rot_alpha * o.confidence, jump_thr);
}

pipepp::pipe_error billiards::pipes::DEPRECATED_marker_finder::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    auto& img           = *i.img_ptr;
    auto  table_rot     = i.table_rot_init;
    auto  table_pos     = i.table_pos_init;
    auto& table_contour = *i.table_contour;

    auto& debug = *i.debug_mat;
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

        auto   outer_contour     = contour;
        auto   inner_contour     = contour;
        auto   m                 = moments(contour);
        auto   center            = Vec2f(m.m10 / m.m00, m.m01 / m.m00);
        double frame_width_outer = table_border_range_outer(ec);
        double frame_width_inner = table_border_range_inner(ec);

        // 각 컨투어의 거리 값에 따라, 차등적으로 밀고 당길 거리를 지정합니다.
        for (int index = 0; index < contour.size(); ++index) {
            Vec2i pt    = contour[index];
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
            pt_dir         = normalize(pt_dir);
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
        auto&        u_hsv = *i.u_hsv;
        vector<UMat> channels;
        split(u_hsv, channels);

        if (channels.size() >= 3) {
            auto&                 u_value = channels[2];
            UMat                  u0, u1;
            UMat                  laplacian_mask(u_hsv.size(), CV_8U, {0});
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
            float rad_min  = marker_area_min_rad(ec);
            float rad_max  = marker_area_max_rad(ec);
            float area_min = marker_area_min_size(ec);
            for (auto& ctr : contours) {
                Point2f center;
                float   radius;
                minEnclosingCircle(ctr, center, radius);
                float area = contourArea(ctr);

                if (area >= area_min && rad_min <= radius && radius <= rad_max) {
                    centers.push_back(center);
                    marker_weights.push_back(contourArea(ctr));
                }
            }
        }
    }

    out.weights = std::move(marker_weights);
    out.markers = std::move(centers);
    return pipepp::pipe_error::ok;
}

void billiards::pipes::DEPRECATED_marker_finder::link_from_previous(shared_data const& sd, table_edge_solver::output_type const& i, input_type& o)
{
    o = input_type{
      .img_ptr        = &sd.imdesc_bkup,
      .img_size       = sd.rgb.size(),
      .table_pos_init = sd.table.pos,
      .table_rot_init = sd.table.rot,
      .debug_mat      = &sd.debug_mat,
      .table_contour  = &sd.table.contour,
      .u_hsv          = &sd.u_hsv,
      .FOV_degree     = sd.camera_FOV(sd)};
}

pipepp::pipe_error billiards::pipes::marker_solver_OLD::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    out.confidence = 0;
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    if (i.p_table_contour == nullptr) {
        return pipepp::pipe_error::warning;
    }

    auto& img           = *i.img_ptr;
    auto  table_rot     = i.table_rot_init;
    auto  table_pos     = i.table_pos_init;
    auto& table_contour = *i.p_table_contour;

    auto& centers        = i.markers;
    auto& marker_weights = i.weights;

    auto& debug = *i.debug_mat;
    if (table_contour.empty() || centers.empty()) {
        return pipepp::pipe_error::ok;
    }

    PIPEPP_ELAPSE_BLOCK("Render Markers")
    {
        Vec2f           fov         = i.FOV_degree;
        constexpr float DtoR        = CV_PI / 180.f;
        auto const      view_planes = generate_frustum(fov[0] * DtoR, fov[1] * DtoR);

        auto& vertexes = i.marker_model;

        auto world_tr = get_transform_matx_fast(table_pos, table_rot);
        for (auto pt : vertexes) {
            Vec4f pt4;
            (Vec3f&)pt4 = pt;
            pt4[3]      = 1.0f;

            pt4 = world_tr * pt4;
            draw_circle(img, (Mat&)debug, 0.01f, (Vec3f&)pt4, {255, 255, 255}, 2);
        }
    }

    PIPEPP_ELAPSE_BLOCK("Solve")
    if (centers.empty() == false) {
        auto& model = i.marker_model;

        struct candidate_t {
            Vec3f  rotation;
            Vec3f  position;
            double suitability;
        };

        vector<candidate_t> candidates;
        candidates.push_back({table_rot, table_pos, 0});

        Vec2f           fov         = i.FOV_degree;
        constexpr float DtoR        = (CV_PI / 180.f);
        auto const      view_planes = generate_frustum(fov[0] * DtoR, fov[1] * DtoR);

        mt19937 rengine(random_device{}());
        float   rotation_variant      = solver::variant_rot(ec);
        float   rotation_axis_variant = solver::variant_rot_axis(ec);
        float   position_variant      = solver::variant_pos(ec);
        int     num_candidates        = solver::num_cands(ec);
        float   error_base            = solver::error_base(ec);
        auto    narrow_rate_pos       = solver::narrow_rate_pos(ec);
        auto    narrow_rate_rot       = solver::narrow_rate_rot(ec);

        auto const& detected = centers;

        for (int iteration = 0, max_iteration = solver::num_iter(ec);
             iteration < max_iteration;
             ++iteration) {
            PIPEPP_ELAPSE_SCOPE_DYNAMIC(fmt::format("Iteration {0:>2}", iteration).c_str());
            auto const& pivot_candidate = candidates.front();

            // candidate 목록 작성.
            // 임의의 방향으로 회전
            // 회전 축은 Y 고정
            while (candidates.size() < num_candidates) {
                candidate_t cand;
                cand.suitability = 0;

                uniform_real_distribution<float> distr_pos{-position_variant, position_variant};
                random_vector(rengine, cand.position, distr_pos(rengine));
                cand.position += pivot_candidate.position;

                uniform_real_distribution<float> distr_rot{-rotation_variant, rotation_variant};
                auto                             rot_norm   = norm(pivot_candidate.rotation);
                auto                             rot_amount = rot_norm + distr_rot(rengine);
                auto                             rotator    = pivot_candidate.rotation / rot_norm;
                random_vector(rengine, cand.rotation, rotation_axis_variant);
                cand.rotation = normalize(cand.rotation + rotator);
                cand.rotation *= rot_amount;

                // 임의의 확률로 180도 회전시킵니다.
                // bool rotate180 = uniform_int_distribution{0, 1}(rengine);
                // if (rotate180) { cand.rotation = rotate_euler(cand.rotation, {0, CV_PI, 0}); }

                candidates.push_back(cand);
            }

            // 평가 병렬 실행
            for_each(execution::par_unseq, candidates.begin(), candidates.end(), [&](candidate_t& elem) {
                thread_local static vector<Vec3f> cc_model;
                thread_local static vector<Vec2f> cc_projected;
                thread_local static vector<Vec2f> cc_detected;

                Vec3f rot = elem.rotation;
                Vec3f pos = elem.position;

                cc_model = model;
                cc_projected.clear();
                cc_detected = detected;
                transform_points_to_camera(img, pos, rot, cc_model);
                project_model_points(img, cc_projected, cc_model, true, view_planes);

                // 각각의 점에 대해 독립적으로 거리를 계산합니다.
                auto suitability = contour_min_dist_for_each(cc_projected, cc_detected, [error_base](float min_dist_sqr) { return pow(error_base, -sqrtf(min_dist_sqr)); });

                elem.suitability = suitability;
            });

            // 가장 confidence가 높은 후보를 선택합니다.
            auto max_it = max_element(execution::par_unseq, candidates.begin(), candidates.end(), [](candidate_t const& a, candidate_t const& b) { return a.suitability < b.suitability; });
            assert(max_it != candidates.end());

            candidates.front() = *max_it;
            candidates.resize(1); // 최적 엘리먼트 제외 모두 삭제
            position_variant *= narrow_rate_pos;
            rotation_variant *= narrow_rate_rot;
        }

        // 디버그 그리기
        auto best = candidates.front();

        PIPEPP_ELAPSE_BLOCK("Render debug glyphs")
        {
            auto          vertexes = model;
            vector<Vec2f> projected;
            transform_points_to_camera(img, best.position, best.rotation, vertexes);
            project_model_points(img, projected, vertexes, true, view_planes);

            for (Point2f pt : detected) {
                line(debug, pt - Point2f(5, 5), pt + Point2f(5, 5), {0, 0, 255}, 2);
                line(debug, pt - Point2f(5, -5), pt + Point2f(5, -5), {0, 0, 255}, 2);
            }

            for (Point2f pt : projected) {
                line(debug, pt - Point2f(5, 0), pt + Point2f(5, 0), {255, 255, 0}, 1);
                line(debug, pt - Point2f(0, 5), pt + Point2f(0, 5), {255, 255, 0}, 1);
            }
        }

        float ampl       = solver::confidence_amp(ec);
        float min_size   = solver::min_valid_marker_size(ec);
        float weight_sum = count_if(marker_weights.begin(), marker_weights.end(), [min_size](float v) { return v > min_size; });
        float apply_rate = min(1.0, ampl * best.suitability / max<double>(1, weight_sum));

        putText(debug, (stringstream() << "marker confidence: " << apply_rate << " (" << best.suitability << "/ " << max<double>(8, detected.size()) << ")").str(), {0, 48}, FONT_HERSHEY_PLAIN, 1.0, {255, 255, 255});

        out.confidence = apply_rate;
        out.table_pos  = best.position;
        out.table_rot  = best.rotation;
    }

    return pipepp::pipe_error::ok;
}

void billiards::pipes::marker_solver_OLD::link_from_previous(shared_data const& sd, DEPRECATED_marker_finder::output_type const& i, input_type& o)
{
    o = input_type{
      .img_ptr         = &sd.imdesc_bkup,
      .img_size        = sd.rgb.size(),
      .table_pos_init  = sd.table.pos,
      .table_rot_init  = sd.table.rot,
      .debug_mat       = &sd.debug_mat,
      .p_table_contour = &sd.table.contour,
      .u_hsv           = &sd.u_hsv,
      .FOV_degree      = sd.camera_FOV(sd),
      .markers         = i.markers,
      .weights         = i.weights};
    sd.get_marker_points_model(o.marker_model);
}

void billiards::pipes::marker_solver_OLD::output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o)
{
    float pos_alpha = shared_data::table::filter::alpha_pos(sd);
    float rot_alpha = shared_data::table::filter::alpha_rot(sd);
    sd.table.pos    = imgproc::set_filtered_table_pos(sd.table.pos, o.table_pos, pos_alpha * o.confidence);
    sd.table.rot    = imgproc::set_filtered_table_rot(sd.table.rot, o.table_rot, rot_alpha * o.confidence);
}

pipepp::pipe_error billiards::pipes::DEPRECATED_ball_search::invoke(pipepp::execution_context& ec, input_type const& input, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    Mat1b table_area(input.img_size, 0);
    auto& table_contour = *input.table_contour;
    auto& sd            = *input.opt_shared;

    auto debug = *input.debug_mat;

    PIPEPP_ELAPSE_BLOCK("Table area mask creation")
    {
        if (table_contour.empty()) { return pipepp::pipe_error::error; }

        vector<Vec2i> pts;
        pts.assign(table_contour.begin(), table_contour.end());
        drawContours(table_area, vector{{pts}}, -1, 255, -1);

        PIPEPP_CAPTURE_DEBUG_DATA_COND((Mat)table_area, show_debug_mat(ec));
    }

    auto& table_pos = input.table_pos;
    auto& table_rot = input.table_rot;

    auto ROI = boundingRect(table_contour);
    if (!get_safe_ROI_rect(debug, ROI)) {
        return pipepp::pipe_error::error;
    }

    const auto u_rgb     = input.u_rgb(ROI);
    const auto u_hsv     = input.u_hsv(ROI);
    auto       area_mask = table_area(ROI);

    vector<cv::UMat> channels;
    split(u_hsv, channels);
    auto [u_h, u_s, u_v] = (cv::UMat(&)[3])(*channels.data());

    cv::UMat u_match_map[3]; // 공의 각 색상에 대한 매치 맵입니다.
    cv::UMat u0, u1;         // 임시 변수 리스트

    // h, s 채널만 사용합니다.
    // 값 형식은 32F이어야 합니다.
    merge(vector{{u_h, u_s}}, u0);
    u0.convertTo(u1, CV_32FC2, 1 / 255.f);
    for (int i = 0; i < 3; ++i) {
        u_match_map[i] = UMat(ROI.size(), CV_32FC2);
        u_match_map[i].setTo(0);
        u1.copyTo(u_match_map[i], area_mask);
    }

    // 각각의 색상에 대해 매칭을 수행합니다.
    auto        imdesc       = *input.imdesc;
    char const* ball_names[] = {"Red", "Orange", "White"};
    float       ball_radius  = shared_data::ball::radius(sd);

    // 테이블 평면 획득
    auto table_plane = plane_t::from_rp(table_rot, table_pos, {0, 1, 0});
    plane_to_camera(imdesc, table_plane, table_plane);
    table_plane.d += matching::cushion_center_gap(ec);

    // 컬러 스케일
    struct ball_desc {
        cv::Vec2f color;
        cv::Vec2f weight;
        double    error_fn_base;
        double    suitability_threshold;
        double    negative_weight;
        double    confidence_threshold;
    } ball_descs[3];

    kangsw::tuple_for_each(
      std::make_tuple(field::red(), field::orange(), field::white()),
      [&]<typename Ty_ = field::red>(Ty_ arg, size_t index) {
          ball_descs[index] = ball_desc{
            .color                 = Ty_::color(ec),
            .weight                = Ty_::weight_hs(ec),
            .error_fn_base         = Ty_::error_fn_base(ec),
            .suitability_threshold = Ty_::suitability_threshold(ec),
            .negative_weight       = Ty_::matching_negative_weight(ec),
            .confidence_threshold  = Ty_::confidence_threshold(ec),
          };
      });

    PIPEPP_ELAPSE_BLOCK("Matching Field Generation")
    for (int ball_idx = 0; ball_idx < 3; ++ball_idx) {
        auto&      m       = u_match_map[ball_idx];
        auto&      bp      = ball_descs[ball_idx];
        cv::Scalar color   = bp.color / 255.f;
        cv::Scalar weight  = bp.weight / norm(bp.weight);
        auto       ln_base = log(bp.error_fn_base);

        cv::subtract(m, color, u0);
        multiply(u0, u0, u1);
        cv::multiply(u1, weight, u0);

        cv::reduce(u0.reshape(1, u0.rows * u0.cols), u1, 1, cv::REDUCE_SUM);
        u1 = u1.reshape(1, u0.rows);
        sqrt(u1, u0);

        multiply(u0, -ln_base, u1);
        exp(u1, m);

        if (show_debug_mat(ec)) {
            PIPEPP_STORE_DEBUG_DATA_DYNAMIC(
              ("Ball Match Field Raw: "s + ball_names[ball_idx]).c_str(),
              m.getMat(ACCESS_FAST).clone());
        }
    }

    cv::UMat match_field{ROI.size(), CV_8UC3};
    match_field.setTo(0);
    cv::Scalar color_ROW[] = {{41, 41, 255}, {0, 213, 255}, {255, 255, 255}};

    // 정규화된 랜덤 샘플의 목록을 만듭니다.
    // 일반적으로 샘플의 아래쪽 반원은 음영에 의해 가려지게 되므로, 위쪽 반원의 샘플을 추출합니다.
    // 이는 정규화된 목록으로, 실제 샘플을 추출할때는 추정 중점 위치에서 계산된 반지름을 곱한 뒤 적용합니다.
    if (ec.consume_option_dirty_flag()) {
        PIPEPP_ELAPSE_SCOPE("Random Sample Generation")

        normal_random_samples_.clear();
        normal_negative_samples_.clear();
        Vec2f  positive_area_range = random_sample::positive_area(ec);
        Vec2f  negative_area_range = random_sample::negative_area(ec);
        int    rand_seed = random_sample::random_seed(ec), circle_radius = random_sample::integral_radius(ec);
        double rotate_angle = random_sample::rotate_angle(ec);

        generate_normalized_sparse_kernel(normal_random_samples_, normal_negative_samples_, positive_area_range, negative_area_range, rand_seed, circle_radius, rotate_angle);

        // 샘플을 시각화합니다.
        if (show_random_sample(ec)) {
            int       scale = random_sample_scale(ec);
            cv::Mat3b random_sample_visualize(scale, scale);
            random_sample_visualize.setTo(0);
            for (auto& pt : normal_random_samples_) {
                random_sample_visualize(cv::Point(pt * scale / 4) + cv::Point{scale / 2, scale / 2}) = {0, 255, 0};
            }
            for (auto& pt : normal_negative_samples_) {
                auto at = Point(pt * scale / 4) + cv::Point{scale / 2, scale / 2};
                if (Rect{{}, random_sample_visualize.size()}.contains(at)) {
                    random_sample_visualize(at) = {0, 0, 255};
                }
            }
            PIPEPP_CAPTURE_DEBUG_DATA((Mat)random_sample_visualize);
        }
    }

    auto const& normal_random_samples   = normal_random_samples_;
    auto const& normal_negative_samples = normal_negative_samples_;

    cv::Mat1f suitability_field{ROI.size()};
    suitability_field.setTo(0);
    pair<vector<cv::Point>, vector<float>> ball_candidates[3];

    array<Point, 4> ball_positions = {};
    array<float, 4> ball_weights   = {};

    PIPEPP_ELAPSE_BLOCK("Color/Edge Matching")
    for (int iter = 3; iter >= 0; --iter) {
        auto      bidx = max(0, iter - 1); // 0, 1 인덱스는 빨간 공 전용
        auto&     m    = u_match_map[bidx];
        cv::Mat1f match;

        auto& bp                 = ball_descs[bidx];
        auto& cand_suitabilities = ball_candidates[bidx].second;
        auto& cand_indexes       = ball_candidates[bidx].first;
        int   min_pixel_radius   = max<int>(1, matching::min_pixel_radius(ec));

        {
            PIPEPP_ELAPSE_SCOPE_DYNAMIC(("Preprocess: "s + ball_names[bidx]).c_str())
            m.copyTo(match);

            // color match값이 threshold보다 큰 모든 인덱스를 선택하고, 인덱스 집합을 생성합니다.
            compare(m, (float)bp.suitability_threshold, u0, cv::CMP_GT);
            bitwise_and(u0, area_mask, u1);

            // 몇 회의 erode 및 dilate 연산을 통해, 중심에 가까운 픽셀을 골라냅니다.
            dilate(u1, u0, {}, Point(-1, -1), matching::num_candidate_dilate(ec));
            erode(u0, u1, {}, Point(-1, -1), matching::num_candidate_erode(ec));
            match_field.setTo(color_ROW[bidx], u1);

            // 모든 valid한 인덱스를 추출합니다.
            cand_indexes.reserve(1000);
            findNonZero(u1, cand_indexes);

            // 인덱스를 임의로 골라냅니다.
            auto num_left = matching::num_maximum_sample(ec);
            // size_t num_left = cand_indexes.size() * (100 - clamp(discard, 0, 100)) / 100;
            discard_random_args(cand_indexes, num_left, mt19937{});

            // 매치 맵의 적합도 합산치입니다.
            cand_suitabilities.resize(cand_indexes.size(), 0);
        }
        float negative_weight = bp.negative_weight;

        // 골라낸 인덱스 내에서 색상 값의 샘플 합을 수행합니다.
        PIPEPP_ELAPSE_SCOPE_DYNAMIC(("Parallel Launch: "s + ball_names[bidx]).c_str());
        auto calculate_suitability = [&](size_t index) {
            auto pt = cand_indexes[index];

            // 현재 추정 위치에서 공의 픽셀 반경 계산
            int ball_pxl_rad = get_pixel_length_on_contact(imdesc, table_plane, pt + ROI.tl(), ball_radius);
            if (ball_pxl_rad < min_pixel_radius) { return; }

            // if 픽셀 반경이 이미지 경계선을 넘어가면 discard
            {
                cv::Point offset{ball_pxl_rad + 1, ball_pxl_rad + 1};
                cv::Rect  image_bound{offset, ROI.size() - (Size)(offset + offset)};

                if (!image_bound.contains(pt)) {
                    return;
                }
            }

            // 각 인덱스에 픽셀 반경을 곱해 매치 맵의 적합도를 합산, 저장
            float suitability = 0;
            {
                float ball_pxl_radf = ball_pxl_rad;

                for (auto roundpt : normal_random_samples) {
                    auto sample_index = pt + cv::Point(roundpt * ball_pxl_radf);
                    suitability += match(sample_index);
                }

                auto bound = ROI;
                bound.x = bound.y = 0;
                for (auto& roundpt : normal_negative_samples) {
                    auto sample_index = pt + Point(roundpt * ball_pxl_radf);
                    if (bound.contains(sample_index)) {
                        suitability -= match(sample_index) * negative_weight;
                    }
                }
            }

            suitability /= normal_random_samples.size();
            suitability_field(pt)     = suitability;
            cand_suitabilities[index] = suitability;
        };

        // 병렬로 launch
        kangsw::iota indexes{cand_indexes.size()};
        if (matching::enable_parallel(ec)) {
            for_each(execution::par_unseq, indexes.begin(), indexes.end(), calculate_suitability);
        } else {
            for_each(execution::seq, indexes.begin(), indexes.end(), calculate_suitability);
        }

        {
            Mat3b debug_ROI = debug(ROI);
            Mat3b adder;
            Mat1b suitability;
            suitability_field.convertTo(suitability, CV_8U, 255);
            //  cv::merge(vector<Mat>{Mat1b::zeros(ROI.size()), suitability, Mat1b::zeros(ROI.size())}, adder);
            cv::merge(vector{suitability, suitability, suitability}, adder);
            debug_ROI -= adder;
        }

        // 특수: 색상이 RED라면 마스크에서 찾아낸 볼에 해당하는 위치를 모두 지우고 위 과정을 다시 수행합니다.
        auto best = max_element(cand_suitabilities.begin(), cand_suitabilities.end());
        if (best == cand_suitabilities.end()) {
            continue;
        }

        if (*best > bp.confidence_threshold) {
            auto best_idx = best - cand_suitabilities.begin();
            auto center   = cand_indexes[best_idx];

            auto rad_px = get_pixel_length_on_contact(imdesc, table_plane, center + ROI.tl(), ball_radius);

            if (rad_px > 0) {
                // 공이 정상적으로 찾아진 경우에만 ...
                // circle(debug, center + ROI.tl(), rad_px, color_ROW[bidx], 1);

                ball_positions[iter] = center;
                ball_weights[iter]   = *best;

                // 빨간 공인 경우 ...
                if (iter == 1) {
                    // Match map에서 검출된 공 위치를 지우고, 위 과정을 반복합니다.
                    circle(m, center, rad_px + field::red::second_ball_erase_radius_adder(ec), 0, -1);
                    PIPEPP_STORE_DEBUG_DATA_COND("Ball Match Field Raw: Red 2", m.getMat(ACCESS_FAST).clone(), show_debug_mat(ec));
                }
            }
        }
    }

    PIPEPP_STORE_DEBUG_DATA("Ball Match Suitability Field", (Mat)suitability_field);
    PIPEPP_STORE_DEBUG_DATA_COND("Ball Match Field Mask", match_field.getMat(ACCESS_FAST).clone(), show_debug_mat(ec));
    for (auto& v : ball_weights) { v *= matching::confidence_weight(ec); }

    // 각 공의 위치를 월드 기준으로 변환합니다.
    array<Vec3f, 4> ballpos;
    for (int i = 0; i < 4; ++i) {
        if (ball_weights[i] == 0) {
            continue;
        }
        auto  pt = ball_positions[i] + ROI.tl();
        Vec3f dst;

        // 공의 중점 방향으로 광선을 투사해 공의 카메라 기준 위치 획득
        dst[0] = pt.x, dst[1] = pt.y, dst[2] = 10;
        get_point_coord_3d(imdesc, dst[0], dst[1], dst[2]);
        auto contact = table_plane.find_contact({}, dst).value();

        Vec3f dummy = {0, 1, 0};
        camera_to_world(imdesc, dummy, contact);
        ballpos[i] = contact;
    }

    // 각 공의 위치를 검토해, 반경보다 작게 검출된 공의 weight를 무효화합니다.
    for (int i = 0; i < 3; ++i) {
        for (int k = i + 1; k < 4; ++k) {
            if (ball_weights[i] == 0) { break; }
            if (ball_weights[k] == 0) { continue; }

            auto distance = norm(ballpos[i] - ballpos[k]);
            if (distance < ball_radius) {
                int min_elem           = ball_weights[i] < ball_weights[k] ? i : k;
                ball_weights[min_elem] = 0;
            }
        }
    }

    PIPEPP_ELAPSE_BLOCK("Correct ball positions")
    {
        // 공의 위치를 이전과 비교합니다.
        ball_position_set descs;
        auto              prev            = input.prev_ball_pos;
        auto              now             = chrono::system_clock::now();
        double            max_error_speed = movement::max_error_speed(ec);

        // 만약 0번 공의 weight가 0인 경우, 즉 공이 하나만 감지된 경우
        // 1번 공의 감지된 위치와 캐시된 0, 1번 공 위치를 비교하고, 1번 공과 더 동떨어진 것을 선택합니다.
        if (ball_weights[0] == 0 && ball_weights[1]) {
            auto p1         = ballpos[1];
            auto diffs      = {norm(p1 - prev[0].pos), norm(p1 - prev[1].pos)};
            auto farther    = distance(diffs.begin(), max_element(diffs.begin(), diffs.end()));
            ball_weights[0] = 0.51f; // magic number ...
            ballpos[0]      = prev[farther].pos;
        }

        // 이전 위치와 비교해, 자리가 바뀐 경우를 처리합니다.
        if (ball_weights[1] && ball_weights[0]) {
            auto p   = ballpos[0],
                 ps0 = prev[0].ps(now),
                 ps1 = prev[1].ps(now);

            if (norm(ps1 - p) < norm(ps0 - p)) {
                swap(ballpos[0], ballpos[1]);
                swap(ball_weights[0], ball_weights[1]);
            }
        }

        double alpha     = movement::alpha_position(ec);
        double jump_dist = movement::jump_distance(ec);
        for (int i = 0; i < 4; ++i) {
            auto& d    = prev[i];
            auto  bidx = std::max(0, i - 1);
            if (ball_weights[i] < ball_descs[bidx].confidence_threshold) {
                continue;
            }

            auto dt          = d.dt(now);
            auto dp          = ballpos[i] - d.pos;
            auto vel_elapsed = dp / dt;

            // 속도 차이가 오차 범위 이내일때만 이를 반영합니다.
            // 속도가 오차 범위를 벗어난 경우 현재 위치와 속도를 갱신하지 않습니다.
            //
            if (norm(vel_elapsed - d.vel) < max_error_speed) {
                // 만약 jump distance보다 위치 변화가 적다면, LPF로 위치를 누적합니다.
                if (norm(dp) < jump_dist) {
                    ballpos[i] = d.pos + (ballpos[i] - d.pos) * alpha;
                }

                descs[i] = ball_position_desc{.pos = ballpos[i], .vel = vel_elapsed, .tp = now};
            } else {
                ball_weights[i] = 0;
            }

            if (ball_weights[i] && show_debug_mat(ec)) {
                draw_circle(imdesc, debug, ball_radius, descs[i].pos, color_ROW[max(i - 1, 0)], 1);
                auto center  = project_single_point(imdesc, descs[i].pos);
                auto pxl_rad = get_pixel_length_on_contact(imdesc, table_plane, center + ROI.tl(), ball_radius);

                putText(debug, to_string(i + 1), center + Point(-7, -pxl_rad), FONT_HERSHEY_PLAIN, 1.3, {0, 0, 0}, 2);
            }
        }

        if (show_debug_mat(ec)) {
            float scale = top_view_scale(ec);

            int  ball_rad_pxl = ball_radius * scale;
            Size total_size   = (Point)(Vec2i)(input.table_outer_size * scale);
            Size fit_size     = (Point)(Vec2i)(input.table_fit_size * scale);
            Size inner_size   = (Point)(Vec2i)(input.table_inner_size * scale);

            Mat3b table_mat(total_size, Vec3b(62, 62, 118));
            rectangle(table_mat, Rect((total_size - fit_size) / 2, fit_size), Scalar{243, 0, 26}, -1);

            Mat3b top_view_mat = table_mat(Rect((total_size - inner_size) / 2, inner_size));
            top_view_mat.setTo(Scalar{196, 74, 86});

            auto inv_tr = get_transform_matx_fast(table_pos, table_rot).inv();
            for (int iter = 0; auto& b : descs) {
                int index = iter++;
                int bidx  = max(0, index - 1);

                if (ball_weights[index] == 0) {
                    continue;
                }

                Vec4f pos4 = (Vec4f&)b.pos;
                pos4(3)    = 1;

                pos4    = inv_tr * pos4;
                auto pt = Point(-pos4[0] * scale, pos4[2] * scale) + (Point)inner_size / 2;

                circle(top_view_mat, pt, ball_rad_pxl, color_ROW[bidx], -1);
                putText(top_view_mat, to_string(iter), pt + Point(-6, 11), FONT_HERSHEY_PLAIN, scale * 0.002, {0, 0, 0}, 2);
            }

            PIPEPP_STORE_DEBUG_DATA("Table Top View", (Mat)table_mat);
        }

        out.new_set = descs;
    }

    return pipepp::pipe_error::ok;
}

void billiards::pipes::DEPRECATED_ball_search::link_from_previous(shared_data const& sd, marker_solver_OLD::output_type const& i, input_type& o)
{
    //o = input_type{
    //  .opt_shared = sd.option(),
    //  .imdesc = &sd.imdesc_bkup,
    //  .debug_mat = &sd.debug_mat,
    //  .prev_ball_pos = sd.state->balls,
    //  .u_rgb = sd.u_rgb,
    //  .u_hsv = sd.u_hsv,
    //  .img_size = sd.rgb.size(),
    //  .table_contour = &sd.table.contour,
    //  .table_pos = sd.state->table.pos,
    //  .table_rot = sd.state->table.rot,
    //  .table_inner_size = shared_data::table::size::inner(sd),
    //  .table_fit_size = shared_data::table::size::fit(sd),
    //  .table_outer_size = shared_data::table::size::outer(sd),
    //};
}

void billiards::pipes::DEPRECATED_ball_search::output_handler(pipepp::pipe_error e, shared_data& sd, output_type const& o)
{
    //if (e == pipepp::pipe_error::ok) {
    //    auto _lck = sd.state->lock();
    //    sd.state->balls = o.new_set;
    //}
}
