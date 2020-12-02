#include "recognizer.hpp"
#include <chrono>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

#include "../image_processing.hpp"
#include "balls.hpp"
#include "fmt/format.h"
#include "marker.hpp"
#include "pipepp/options.hpp"
#include "pipepp/pipe.hpp"
#include "table_search.hpp"
#include "traditional.hpp"

#pragma warning(disable : 4244)

namespace billiards::pipes {
namespace {

struct marker_search_to_solve {
    PIPEPP_DECLARE_OPTION_CLASS(table_marker_finder);

    static void link(
      shared_data&                            sd,
      table_marker_finder::output_type const& o,
      marker_solver_cpu::input_type&          i)
    {
        if (o.marker_weight_map.empty()) {
            i.p_table_contour = nullptr;
            return;
        }

        i = marker_solver_cpu::input_type{
          .img_ptr         = &sd.imdesc_bkup,
          .img_size        = sd.rgb.size(),
          .table_pos_init  = sd.table.pos,
          .table_rot_init  = sd.table.rot,
          .debug_mat       = &sd.debug_mat,
          .p_table_contour = &sd.table.contour,
          .u_hsv           = &sd.u_hsv,
          .FOV_degree      = sd.camera_FOV(sd)};
        sd.get_marker_points_model(i.marker_model);

        // Weight map으로부터 마커 목록 찾기
        using namespace std;
        using namespace cv;
        Mat1b mat = o.marker_weight_map > 0.02f;
        Mat1b expand;
        dilate(mat, expand, {});

        vector<vector<Vec2i>> contours;
        findContours(expand - mat, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (auto& contour : contours) {
            auto m = moments(contour);
            if (m.m00) {
                Vec2f centerf(m.m10 / m.m00, m.m01 / m.m00);
                Point center = (Vec2i)centerf;
                i.markers.push_back(centerf);
                i.weights.push_back(o.marker_weight_map(center));
            }
        }
    }
};

} // namespace
} // namespace billiards::pipes

namespace billiards {
class recognizer_t;
}

void build_traditional_path(pipepp::pipeline<billiards::pipes::shared_data, billiards::pipes::input_resize>::initial_proxy_type input_proxy, pipepp::pipe_proxy<billiards::pipes::shared_data, billiards::pipes::output_pipe> output_pipe_proxy);

auto billiards::pipes::build_pipe() -> std::shared_ptr<pipepp::pipeline<shared_data, input_resize>>
{
    auto pl = decltype(build_pipe())::element_type::make(
      "input", 1, &pipepp::make_executor<pipes::input_resize>);

    // INPUT
    auto input_proxy = pl->front();
    input_proxy.add_output_handler(&input_resize::output_handler);
    input_proxy.configure_tweaks().selective_output = true;

    auto output_pipe_proxy = pl->create("output", 1, &output_pipe::factory);

    // ---------------------------------------------------------------------------------
    //  INPUT --> [PREPROCESSOR]
    // ---------------------------------------------------------------------------------
    //      CASE A: TRADITIONAL CONTOUR SEARCH
    // ALL PIPES
    {
        auto contour_search_proxy  = pl->create("table contour searcher", 1, &pipepp::make_executor<table_contour_geometric_search>);
        auto pnp_solver_proxy      = pl->create("solve table edge", 1, &pipepp::make_executor<table_edge_solver>);
        auto marker_search_proxy   = pl->create("table marker search", 1, &pipepp::make_executor<table_marker_finder>);
        auto marker_solver_proxy   = pl->create("table marker solver", 1, &pipepp::make_executor<marker_solver_cpu>);
        auto marker_solver_2_proxy = pl->create("table marker solver gpu", 1, &pipepp::make_executor<marker_solver_gpu>);

        input_proxy.link_output(contour_search_proxy, &table_contour_geometric_search_link);

        contour_search_proxy.add_output_handler(
          [](shared_data&                                       sd,
             table_contour_geometric_search::output_type const& o) {
              sd.table.contour = o.contours;
          });
        contour_search_proxy.link_output(pnp_solver_proxy, &table_edge_solver::link_from_previous);

        pnp_solver_proxy.add_output_handler(&table_edge_solver::output_handler);
        pnp_solver_proxy.link_output(marker_search_proxy, &table_marker_finder::link);

        marker_search_proxy.link_output(marker_solver_proxy, &marker_search_to_solve::link);
        marker_search_proxy.link_output(marker_solver_2_proxy, &marker_solver_gpu::link);

        marker_solver_proxy.add_output_handler(&marker_solver_cpu::output_handler);
        for (auto ballidx : kangsw::counter(3)) {
            constexpr char const* BALL_NAME[]  = {"Red", "Orange", "White"};
            constexpr int         IDX_OFFSET[] = {0, 2, 3};
            auto                  ball_proxy   = pl->create(fmt::format("ball finder: {}", BALL_NAME[ballidx]),
                                         2, &pipepp::make_executor<ball_finder_executor>);

            // 이전 출력 연결
            marker_solver_proxy.link_output(ball_proxy, &ball_finder_executor::link);

            // 빨간 공이 2개이므로, 인덱스를 각각 0, 2, 3 부여합니다.
            ball_proxy.add_output_handler(
              [ball_index_offset = IDX_OFFSET[ballidx]] //
              (shared_data & sd,
               ball_finder_executor::output_type const& o) //
              {
                  for (auto  idx = ball_index_offset;
                       auto& pt : o.positions) //
                  {
                      sd.update_ball_pos(idx++, pt.position, pt.confidence);
                  }
              });

            // 출력 연결
            ball_proxy.link_output(output_pipe_proxy, &output_pipe::link_from_previous);
        }
    }

    return pl;

    //      CASE A: SUPERPIXELS
    auto superpixels_proxy
      = pl->create("superpixel", std::thread::hardware_concurrency() / 2, &pipepp::make_executor<superpixel_executor>);
    input_proxy.link_output(superpixels_proxy, &superpixel_executor::link_from_previous);
    superpixels_proxy.add_output_handler(&superpixel_executor::output_handler);
    superpixels_proxy.configure_tweaks().selective_output = true;

    // ---------------------------------------------------------------------------------
    auto clustering_proxy
      = superpixels_proxy.create_and_link_output(
        "clustering",
        std::thread::hardware_concurrency() / 2,
        &kmeans_executor::link_from_previous,
        &pipepp::make_executor<kmeans_executor>);
    clustering_proxy.add_output_handler(&kmeans_executor::output_handler);

    auto edge_finder_0_proxy
      = clustering_proxy.create_and_link_output(
        "edge finder 0",
        1,
        [](shared_data& sd, kmeans_executor::output_type const& o, label_edge_detector::input_type& i) {
            if (sd.cluster.label_2d_spxl.empty() || sd.cluster.label_cluster_1darray.empty()) {
                return false;
            }
            i.labels = imgproc::index_by(sd.cluster.label_cluster_1darray, sd.cluster.label_2d_spxl);
            return true;
        },
        &pipepp::make_executor<label_edge_detector>);

    auto table_contour_finder_proxy
      = edge_finder_0_proxy.create_and_link_output(
        "table contour search",
        1,
        [](shared_data& sd, label_edge_detector::output_type const& o, hough_line_executor::input_type& i) {
            i.edges   = o.edges;
            i.dbg_mat = &sd.debug_mat;
        },
        &pipepp::make_executor<hough_line_executor>);

    build_traditional_path(input_proxy, output_pipe_proxy);
    table_contour_finder_proxy.link_output(output_pipe_proxy, &output_pipe::link_from_previous);

    return pl;
}

cv::Mat billiards::pipes::shared_data::retrieve_image_in_colorspace(kangsw::hash_index hash)
{
    int _ph0, to;
    imgproc::color_space_to_flag(hash, to, _ph0);

    std::unique_lock _lck{lock_};
    if (to == -1) { return rgb; }
    auto [it, should_generate] = converted_resources_.try_emplace(hash);

    if (should_generate) {
        cv::cvtColor(rgb, it->second, to);
    }

    return it->second;
}

void billiards::pipes::shared_data::store_image_in_colorspace(kangsw::hash_index hash, cv::Mat v)
{
    std::unique_lock _lck{lock_};
    converted_resources_.emplace(hash, std::move(v));
}

void billiards::pipes::shared_data::update_ball_pos(size_t ball_idx, cv::Vec3f pos, float conf)
{
    if (ball_idx >= std::size(balls_)) { return; }
    // 공의 좌표를 테이블 기준으로 변환해서 저장.
    using namespace imgproc;
    auto tr_inv = imgproc::get_transform_matx_fast(table.pos, table.rot).inv();
    pos         = subvec<0, 3>(tr_inv * concat_vec(pos, 1.f));

    auto& [elem, confidence] = balls_[ball_idx];
    elem.tp                  = launch_time_point();
    elem.pos                 = pos;
    confidence               = conf;
}

billiards::pipes::ball_position_desc billiards::pipes::shared_data::get_ball(size_t bidx) const
{
    auto ball = balls_[bidx].second > 0 ? balls_[bidx].first : state_->balls[bidx];

    // 공의 좌표를 월드 기준으로 변환 후 내보냄
    using namespace imgproc;
    auto tr  = get_transform_matx_fast(table.pos, table.rot);
    ball.pos = subvec<0, 3>(tr * concat_vec(ball.pos, 1.f));
    return ball;
}

auto billiards::pipes::shared_data::get_ball_raw(size_t bidx) const -> ball_position_desc
{
    return balls_[bidx].second > 0 ? balls_[bidx].first : state_->balls[bidx];
}

void billiards::pipes::shared_data::_on_all_ball_gathered()
{
    using namespace cv;
    using namespace std;
    using kangsw::counter;

}

pipepp::pipe_error billiards::pipes::input_resize::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto src_size     = i.rgba.size();
    auto width        = std::min(src_size.width, desired_image_width(ec));
    auto height       = int((int64_t)width * src_size.height / src_size.width);
    out.resized_scale = width / (float)src_size.width;

    PIPEPP_ELAPSE_BLOCK("RGBA to RGB")
    {
        cv::Mat rgb;

        cv::cvtColor(i.rgba, rgb, cv::COLOR_BGRA2RGB);
        if (width < src_size.width) {
            PIPEPP_ELAPSE_SCOPE("Resizing");
            cv::resize(rgb, out.rgb, {width, height});
        } else {
            out.rgb = std::move(rgb);
        }
        out.rgb.copyTo(out.u_rgb);
    }

    PIPEPP_ELAPSE_BLOCK("Color convert")
    {
        cv::cvtColor(out.u_rgb, out.u_hsv, cv::COLOR_RGB2HSV);
        out.u_hsv.copyTo(out.hsv);
    }

    if (show_sources(ec)) {
        PIPEPP_ELAPSE_SCOPE("Debug data generation");

        PIPEPP_STORE_DEBUG_DATA("Source RGB", out.rgb.clone());
        PIPEPP_STORE_DEBUG_DATA("Source HSV", out.hsv.clone());
    }

    if (test_color_spaces(ec)) {
        auto testcode = [&](const char* head, const char* tail, int color_space) {
            PIPEPP_ELAPSE_SCOPE_DYNAMIC(fmt::format("Color Space: {}", head).c_str());

            cv::Mat converted;
            cv::Mat ch[3];
            cv::cvtColor(out.rgb, converted, color_space);
            cv::split(converted, ch);

            PIPEPP_STORE_DEBUG_DATA_DYNAMIC(fmt::format("{}", head).c_str(), converted);
            for (int idx = 0; tail[idx] && idx < 3; ++idx) {
                PIPEPP_STORE_DEBUG_DATA_DYNAMIC(fmt::format("{}: {}", head, tail[idx]).c_str(), ch[idx]);
            }
        };

        testcode("LAB", "LAB", cv::COLOR_RGB2Lab);
        testcode("Luv", "Luv", cv::COLOR_RGB2Luv);
        testcode("YUV", "YUV", cv::COLOR_RGB2YUV);
        testcode("HLS", "HLS", cv::COLOR_RGB2HLS);
        testcode("YCrCb", "YRB", cv::COLOR_RGB2YCrCb);
        testcode("XYZ", "XYZ", cv::COLOR_RGB2XYZ);

        {
            PIPEPP_ELAPSE_SCOPE("RGB, HSV")
            cv::Mat ch[3];
            cv::split(out.hsv, ch);
            PIPEPP_STORE_DEBUG_DATA("HSV: H", ch[0].clone());
            PIPEPP_STORE_DEBUG_DATA("HSV: S", ch[1].clone());
            PIPEPP_STORE_DEBUG_DATA("HSV: V", ch[2].clone());

            cv::split(out.rgb, ch);
            PIPEPP_STORE_DEBUG_DATA("RGB: R", ch[0]);
            PIPEPP_STORE_DEBUG_DATA("RGB: G", ch[1]);
            PIPEPP_STORE_DEBUG_DATA("RGB: B", ch[2]);
        }
    }

    out.img_size = cv::Size(width, height);

    return pipepp::pipe_error::ok;
}

void billiards::pipes::input_resize::output_handler(pipepp::pipe_error, shared_data& sd, pipepp::execution_context& ec, output_type const& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);

    PIPEPP_ELAPSE_BLOCK("Data Copy & Generation")
    {
        o.u_hsv.copyTo(sd.u_hsv);
        o.u_rgb.copyTo(sd.u_rgb);
        o.rgb.copyTo(sd.rgb);
        o.hsv.copyTo(sd.hsv);

        o.rgb.copyTo(sd.debug_mat);

        for (auto& c = sd.imdesc_bkup.camera;
             auto  p : {&c.fx, &c.fy, &c.cx, &c.cy}) //
        {
            *p *= o.resized_scale;
        }

        using namespace kangsw::literals;
        sd.store_image_in_colorspace("HSV"_hash, sd.hsv);
    }

    PIPEPP_ELAPSE_BLOCK("Table Area Edge Detection")
    {
        imgproc::range_filter(
          o.hsv,
          sd.table_hsv_filtered,
          shared_data::table::filter::color_lo(sd),
          shared_data::table::filter::color_hi(sd));

        imgproc::carve_outermost_pixels(sd.table_hsv_filtered, 0);
        erode(sd.table_hsv_filtered, sd.table_filtered_edge, {});
        sd.table_filtered_edge = sd.table_hsv_filtered - sd.table_filtered_edge;

        if (input_resize::show_sources(ec)) {
            PIPEPP_STORE_DEBUG_DATA("Filtered mask", (cv::Mat)sd.table_hsv_filtered);
        }
    }
}

pipepp::pipe_error billiards::pipes::output_pipe::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto& sd     = *i;
    auto& debug  = sd.debug_mat;
    auto& imdesc = sd.imdesc_bkup;
    using namespace imgproc;
    using namespace std;
    using namespace cv;

    // 다 모인 공의 위치를 처리합니다.
    sd._on_all_ball_gathered();

    if (debug::render_debug_glyphs(ec)) {
        { // Draw table info
            auto&         pos = sd.table.pos;
            auto&         rot = sd.table.rot;
            auto          FOV = sd.camera_FOV(sd);
            vector<Vec3f> model;
            get_table_model(model, shared_data::table::size::fit(sd));
            project_contours(imdesc, debug, model, pos, rot, {0, 255, 0}, 1, FOV);
            get_table_model(model, shared_data::table::size::inner(sd));
            project_contours(imdesc, debug, model, pos, rot, {83, 0, 213}, 1, FOV);
            get_table_model(model, shared_data::table::size::outer(sd));
            project_contours(imdesc, debug, model, pos, rot, {221, 64, 0}, 1, FOV);

            draw_axes(imdesc, debug, rot, pos, 0.1, 2);
        }

        { // Draw ball info
            float scale = debug::table_top_view_scale(ec);

            int  ball_rad_pxl = shared_data::ball::radius(sd) * scale;
            Size total_size   = (Point)(Vec2i)(shared_data::table::size::outer(sd) * scale);
            Size fit_size     = (Point)(Vec2i)(shared_data::table::size::fit(sd) * scale);
            Size inner_size   = (Point)(Vec2i)(shared_data::table::size::inner(sd) * scale);

            Mat3b table_mat(total_size, Vec3b(118, 62, 62));
            rectangle(table_mat, Rect((total_size - fit_size) / 2, fit_size), Scalar{26, 0, 243}, -1);

            Mat3b top_view_mat = table_mat(Rect((total_size - inner_size) / 2, inner_size));
            top_view_mat.setTo(Scalar{86, 74, 195});

            auto      inv_tr      = get_transform_matx_fast(sd.table.pos, sd.table.rot).inv();
            cv::Vec3b color_ROW[] = {{231, 11, 11}, {231, 106, 2}, {241, 241, 211}};
            for (auto index : kangsw::counter(4)) {
                int  bidx = max(0, index - 1);
                auto b    = sd.get_ball(index);

                Vec4f pos4 = imgproc::concat_vec(b.pos, 1.f);

                pos4    = inv_tr * pos4;
                auto pt = Point(-pos4[0] * scale, pos4[2] * scale) + (Point)inner_size / 2;

                circle(top_view_mat, pt, ball_rad_pxl, color_ROW[bidx], -1);
                putText(top_view_mat, to_string(index), pt + Point(-6, 11), FONT_HERSHEY_PLAIN, scale * 0.002, {0, 0, 0}, 2);
            }

            PIPEPP_STORE_DEBUG_DATA("Table Top View", (Mat)table_mat);
        }
    }

    PIPEPP_STORE_DEBUG_DATA("Debug glyphs rendering", sd.debug_mat.clone());

    PIPEPP_ELAPSE_BLOCK("Output callback execution")
    {
        // 출력 전, 테이블 위치 오프셋

        nlohmann::json output_to_unity;
        using std::chrono::duration;
        {
            auto& desc       = output_to_unity;
            using table_size = shared_data::table::size;

            if (auto now = clock::now();
                duration<float>(now - latest_setting_refresh_).count()
                  > legacy::setting_refresh_interval(ec)
                || ec.consume_option_dirty_flag()) //
            {
                latest_setting_refresh_ = now;
                using table_filter      = shared_data::table::filter;

                Vec3f tf[2]     = {table_filter::color_lo(sd), table_filter::color_hi(sd)};
                auto [min, max] = tf;
                enum { H,
                       S,
                       V };

                // -- 시작 파라미터 전송
                auto inner                        = table_size::inner(sd);
                desc["TableProps"]["InnerWidth"]  = inner[0];
                desc["TableProps"]["InnerHeight"] = inner[1];

                using unity                                          = legacy::unity;
                desc["TableProps"]["EnableShaderApplyDepthOverride"] = unity::enable_table_depth_override(ec);
                desc["TableProps"]["ShaderMinH"]                     = min[H] * (1 / 180.f);
                desc["TableProps"]["ShaderMaxH"]                     = max[H] * (1 / 180.f);
                desc["TableProps"]["ShaderMinS"]                     = min[S] * (1 / 255.f);
                desc["TableProps"]["ShaderMaxS"]                     = max[S] * (1 / 255.f);

                using phys                         = unity::phys;
                desc["BallRadius"]                 = phys::simulation_ball_radius(ec);
                desc["Phys"]["BallRestitution"]    = phys::ball_restitution(ec);
                desc["Phys"]["BallDamping"]        = phys::velocity_damping(ec);
                desc["Phys"]["BallStaticFriction"] = phys::roll_coeff_on_contact(ec);
                desc["Phys"]["BallRollTime"]       = phys::roll_begin_time(ec);
                desc["Phys"]["TableRestitution"]   = phys::table_restitution(ec);
                desc["Phys"]["TableRtoVCoeff"]     = phys::table_roll_to_velocity_coeff(ec);
                desc["Phys"]["TableVtoRCoeff"]     = phys::table_velocity_to_roll_coeff(ec);

                desc["CameraAnchorOffset"] = unity::anchor_offset_vector(ec);
            }
            auto ofst = legacy::unity::table_output_offset(ec);
            ofst      = rodrigues(sd.table.rot) * ofst;

            desc["Table"]["Translation"] = sd.table.pos + ofst;
            desc["Table"]["Orientation"] = sd.table.rot;
            desc["Table"]["Confidence"]  = sd.table.confidence;

            char const* ball_names[] = {"Red1", "Red2", "Orange", "White"};
            for (auto idx : kangsw::counter(4)) {
                auto ball = sd.get_ball(idx);
                auto conf = sd.get_ball_conf(idx);

                auto bname                = ball_names[idx];
                desc[bname]["Position"]   = ball.pos;
                desc[bname]["Confidence"] = conf;
            }
        }
        sd.callback(sd.imdesc_bkup, output_to_unity);
    }
    return pipepp::pipe_error::ok;
}

void billiards::pipes::shared_data::get_marker_points_model(std::vector<cv::Vec3f>& model) const
{
    auto& ec = *this;

    int   num_x                = table::marker::count_x(ec);
    int   num_y                = table::marker::count_y(ec);
    float felt_width           = table::marker::felt_width(ec);
    float felt_height          = table::marker::felt_height(ec);
    float dist_from_felt_long  = table::marker::dist_from_felt_long(ec);
    float dist_from_felt_short = table::marker::dist_from_felt_short(ec);
    float step                 = table::marker::step(ec);
    float width_shift_a        = table::marker::width_shift_a(ec);
    float width_shift_b        = table::marker::width_shift_b(ec);
    float height_shift_a       = table::marker::height_shift_a(ec);
    float height_shift_b       = table::marker::height_shift_b(ec);

    for (int i = -num_y / 2; i < num_y / 2 + 1; ++i) {
        model.emplace_back(-(dist_from_felt_short + felt_width / 2), 0, step * i + height_shift_a);
        model.emplace_back(+(dist_from_felt_short + felt_width / 2), 0, step * -i + height_shift_b);
    }
    for (int i = -num_x / 2; i < num_x / 2 + 1; ++i) {
        model.emplace_back(step * i + width_shift_a, 0, -(dist_from_felt_long + felt_height / 2));
        model.emplace_back(step * -i + width_shift_b, 0, +(dist_from_felt_long + felt_height / 2));
    }
}