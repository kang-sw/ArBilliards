#include "recognizer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

#include "../image_processing.hpp"
#include "fmt/format.h"
#include "marker.hpp"
#include "pipepp/options.hpp"
#include "table_search.hpp"
#include "traditional.hpp"

#pragma warning(disable : 4244)

namespace billiards::pipes {
namespace {

struct marker_search_to_solve {
    PIPEPP_DECLARE_OPTION_CLASS(table_marker_finder);

    static bool link(
      shared_data& sd,
      table_marker_finder::output_type const& o,
      marker_solver_OLD::input_type& i)
    {
        i = marker_solver_OLD::input_type{
          .img_ptr = &sd.imdesc_bkup,
          .img_size = sd.rgb.size(),
          .table_pos_init = sd.table.pos,
          .table_rot_init = sd.table.rot,
          .debug_mat = &sd.debug_mat,
          .table_contour = &sd.table.contour,
          .u_hsv = &sd.u_hsv,
          .FOV_degree = sd.camera_FOV(sd)};
        sd.get_marker_points_model(i.marker_model);

        // Weight map으로부터 마커 목록 찾기
        using namespace std;
        using namespace cv;
        Mat1b mat = o.marker_weight_map > 0.1f;
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

        return i.markers.empty() == false;
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
    output_pipe_proxy.configure_tweaks().selective_input = true;

    // ---------------------------------------------------------------------------------
    //  INPUT --> [PREPROCESSOR]
    // ---------------------------------------------------------------------------------
    //      CASE A: TRADITIONAL CONTOUR SEARCH
    // ALL PIPES
    {
        auto contour_search_proxy = pl->create("table contour searcher", 1, &pipepp::make_executor<table_contour_geometric_search>);
        auto pnp_solver_proxy = pl->create("solve table edge", 1, &pipepp::make_executor<table_edge_solver>);
        auto marker_search_proxy = pl->create("table marker search", 1, &pipepp::make_executor<table_marker_finder>);
        auto marker_solver_proxy = pl->create("table marker solver", 1, &pipepp::make_executor<marker_solver_OLD>);

        input_proxy.link_output(contour_search_proxy,
                                [](shared_data& sd, table_contour_geometric_search::input_type& i) {
                                    i.debug_rgb = &(cv::Mat3b&)sd.debug_mat;
                                    i.edge_img = sd.table_filtered_edge;
                                });

        contour_search_proxy.add_output_handler(
          [](shared_data& sd,
             table_contour_geometric_search::output_type const& o) {
              sd.table.contour = o.contours;
          });
        contour_search_proxy.link_output(pnp_solver_proxy, &table_edge_solver::link_from_previous);

        pnp_solver_proxy.add_output_handler(&table_edge_solver::output_handler);
        pnp_solver_proxy.link_output(marker_search_proxy, &table_marker_finder::link);

        marker_search_proxy.link_output(marker_solver_proxy, &marker_search_to_solve::link);

        marker_solver_proxy.link_output(output_pipe_proxy, &output_pipe::link_from_previous);
        marker_solver_proxy.add_output_handler(&marker_solver_OLD::output_handler);
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
            i.edges = o.edges;
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

    if (to == -1) { return rgb; }
    auto [it, should_generate] = converted_resources_.try_emplace(hash);

    if (should_generate) {
        cv::cvtColor(rgb, it->second, to);
    }

    return it->second;
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
        if (width < src_size.width) {
            cv::resize(rgb, out.u_rgb, {width, height});
        } else {
            out.u_rgb = std::move(rgb);
        }
    }

    PIPEPP_ELAPSE_BLOCK("Color convert")
    {
        out.u_rgb.copyTo(out.rgb);
        cv::cvtColor(out.u_rgb, out.u_hsv, cv::COLOR_RGB2HSV);
        out.u_hsv.copyTo(out.hsv);
    }

    if (show_sources(ec)) {
        PIPEPP_ELAPSE_SCOPE("Debug data generation");

        PIPEPP_STORE_DEBUG_DATA("Source RGB", out.rgb.clone());
        PIPEPP_STORE_DEBUG_DATA("Source HSV", out.hsv.clone());

        cv::Mat ch[3];
        cv::split(out.hsv, ch);
        PIPEPP_STORE_DEBUG_DATA("HSV: H", ch[0]);
        PIPEPP_STORE_DEBUG_DATA("HSV: S", ch[1]);
        PIPEPP_STORE_DEBUG_DATA("HSV: V", ch[2]);
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
    }
}

pipepp::pipe_error billiards::pipes::output_pipe::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto& sd = *i;
    auto& debug = sd.debug_mat;
    auto& imdesc = sd.imdesc_bkup;
    using namespace imgproc;
    using namespace std;
    using namespace cv;

    {
        auto _lck = sd.state_->lock();
        sd.state_->table.pos = sd.table.pos;
        sd.state_->table.rot = sd.table.rot;
    }

    if (debug::render_debug_glyphs(ec)) {
        auto& pos = sd.table.pos;
        auto& rot = sd.table.rot;
        auto FOV = sd.camera_FOV(sd);
        vector<Vec3f> model;
        get_table_model(model, shared_data::table::size::fit(sd));
        project_contours(imdesc, debug, model, pos, rot, {0, 255, 0}, 2, FOV);
        get_table_model(model, shared_data::table::size::inner(sd));
        project_contours(imdesc, debug, model, pos, rot, {221, 64, 0}, 2, FOV);
        get_table_model(model, shared_data::table::size::outer(sd));
        project_contours(imdesc, debug, model, pos, rot, {83, 0, 213}, 1, FOV);

        draw_axes(imdesc, debug, pos, rot, 0.1, 2);
    }

    PIPEPP_STORE_DEBUG_DATA("Debug glyphs rendering", sd.debug_mat.clone());
    return pipepp::pipe_error::ok;
}