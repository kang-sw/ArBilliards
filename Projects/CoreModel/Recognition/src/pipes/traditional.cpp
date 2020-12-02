#include "traditional.hpp"

#include "../image_processing.hpp"
#include "fmt/format.h"

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

            cv::Vec3f rv = rot, tv = pos;

            camera_to_world(img, rv, tv);
            world_to_camera(img, rv, tv);
            rot = rv, pos = tv;

            // confidence 계산
            auto vertexes = obj_pts;
            for (auto& vtx : vertexes) {
                vtx = rodrigues(rot) * vtx + pos;
            }

            PIPEPP_ELAPSE_SCOPE("Projection");
            vector<Vec2f> mapped;
            project_model_local(img, mapped, vertexes, false, {});

            // 각 점을 비교하여 에러를 계산합니다.
            double error_sum = 0;
            for (size_t index = 0; index < 4; index++) {
                Vec2f projpt = mapped[index];
                error_sum += norm(projpt - table_contour[index], NORM_L2SQR);
            }

            PIPEPP_STORE_DEBUG_DATA_DYNAMIC((std::to_string(iter) + " iteration error").c_str(), error_sum);
            auto conf = pow(pnp_error_exp_fn_base(ec), -sqrt(error_sum));
            if (conf > max_confidence) {
                max_confidence = conf, tvec = pos, rvec = rot;
            }

            {
                vector<Vec2i> pp;
                pp.assign(mapped.begin(), mapped.end());
                drawContours(i.debug_mat, vector{{pp}}, -1, {255, 65, 65}, 3);
            }

            // 점 배열 1개 회전
            if (iter == 0) {
                table_contour.push_back(table_contour.front());
                table_contour.erase(table_contour.begin());
            }
        }

        if (max_confidence > pnp_conf_threshold(ec)) {
            camera_to_world(img, rvec, tvec);

            o.confidence = max_confidence;
            o.table_pos  = tvec;
            o.table_rot  = rvec;
            o.can_jump   = true;
            draw_axes(img, i.debug_mat, rvec, tvec, 0.05f, 3);

            PIPEPP_STORE_DEBUG_DATA("Full PNP data confidence", o.confidence);
            PIPEPP_STORE_DEBUG_STR("RVEC", rvec);
        } else {
            o.confidence = 0;
        }
    }

    if (o.confidence > 0.1) {
        using namespace cv;
        PIPEPP_ELAPSE_SCOPE("Visualize");
        auto& rvec = o.table_rot;
        auto& tvec = o.table_pos;
        draw_axes(img, i.debug_mat, rvec, tvec, 0.05f, 3);
        Scalar color = o.can_jump ? Scalar{255, 0, 255} : Scalar{0, 255, 255};
        project_contours(img, i.debug_mat, obj_pts, tvec, rvec, color, 3, {86, 58});
    }

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
    // 항상 출력의 X축을 뒤집습니다.
    auto out_rot = imgproc::rotate_euler(o.table_rot, {0, 0, (float)CV_PI});

    if (o.confidence < 1e-6f) {
        // edge solver는 jump가 아니라면 rotation을 손대지 않습니다.
        sd.table.confidence = 0;
        return;
    }
    sd.table.confidence = o.confidence;
    sd.table.pos        = o.table_pos;
    sd.table.rot        = imgproc::set_filtered_table_rot(sd.table.rot, out_rot, 1);
}
