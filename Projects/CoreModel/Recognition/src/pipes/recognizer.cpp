#include "recognizer.hpp"

#include <random>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include "fmt/format.h"
#include "table_search.hpp"
#include "../image_processing.hpp"
#include "pipepp/options.hpp"
#include "traditional.hpp"

#pragma warning(disable : 4244)

namespace billiards
{
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
    //      CASE A: SUPERPIXELS
    auto superpixels_proxy
      = input_proxy.create_and_link_output(
        "superpixel",
        std::thread::hardware_concurrency() / 2,
        &superpixel_executor::link_from_previous,
        &pipepp::make_executor<superpixel_executor>);
    superpixels_proxy.add_output_handler(&superpixel_executor::output_handler);
    superpixels_proxy.configure_tweaks().selective_output = true;

    // ---------------------------------------------------------------------------------
    //          SUPERPIXELS< CONTOUR SEARCH
    //
    {
        auto table_contour_finder_2_proxy = pl->create(
          "table contour search 2", 4, &pipepp::make_executor<hough_line_executor>);
        superpixels_proxy
          .link_output(table_contour_finder_2_proxy,
                       [](shared_data const& sd, hough_line_executor::input_type& i) {
                           i.dbg_mat = &sd.debug_mat;
                           auto& mask = sd.table_hsv_filtered;
                           cv::Mat eroded;
                           cv::erode(mask, eroded, {});
                           i.edges = mask - eroded;
                       });
    }

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

    o.u_hsv.copyTo(sd.u_hsv);
    o.u_rgb.copyTo(sd.u_rgb);
    o.rgb.copyTo(sd.rgb);
    o.hsv.copyTo(sd.hsv);

    o.rgb.copyTo(sd.debug_mat);

    imgproc::filter_hsv(
      o.hsv,
      sd.table_hsv_filtered,
      shared_data::table::filter::color_lo(sd),
      shared_data::table::filter::color_hi(sd));

    PIPEPP_STORE_DEBUG_DATA("Filtered HSV", (cv::Mat)sd.table_hsv_filtered);
}

pipepp::pipe_error billiards::pipes::output_pipe::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto& sd = *i;

    PIPEPP_STORE_DEBUG_DATA("Debug glyphs rendering", sd.debug_mat.clone());
    return pipepp::pipe_error::ok;
}