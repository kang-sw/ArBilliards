#include "recognizer.hpp"

#include <opencv2/imgproc.hpp>

auto billiards::pipes::build_pipe() -> std::shared_ptr<pipepp::pipeline<shared_data, input_resize>>
{
    auto pl = decltype(build_pipe())::element_type::create(
      "input", 1, &pipepp::make_executor<pipes::input_resize>);

    auto input = pl->front();
    input.add_output_handler(&input_resize::output_handler);
    auto contour_search = input.create_and_link_output("contour search", false, 1, &contour_candidate_search::link_from_previous, &pipepp::make_executor<contour_candidate_search>);

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
        auto eps0 = contour_approx_epsilon_preprocess(ec);
        auto eps1 = contour_approx_epsilon_convex_hull(ec);
        auto size_threshold = contour_area_threshold_ratio(ec) * image_size.area();

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

        PIPEPP_STORE_DEBUG_DATA("Debug Mat", debug);
    }

    o.table_contour_candidate = move(table_contour);
    return pipepp::pipe_error::ok;
}

void billiards::pipes::contour_candidate_search::link_from_previous(shared_data const& sd, input_resize::output_type const& i, input_type& o)
{
    o.u_hsv = i.u_hsv;
    o.debug_display = sd.debug_mat;
}

pipepp::pipe_error billiards::pipes::table_edge_solver::invoke(pipepp::execution_context& ec, input_type const& i, output_type& out)
{
    return {};
}

void billiards::imgproc::filter_hsv(cv::InputArray input, cv::OutputArray output, cv::Vec3f min_hsv, cv::Vec3f max_hsv)
{
    using namespace cv;
    if (max_hsv[0] < min_hsv[0]) {
        UMat mask, hi, lo, temp;
        auto filt_min = min_hsv, filt_max = max_hsv;
        filt_min[0] = 0, filt_max[0] = 255;

        inRange(input, filt_min, filt_max, mask);
        inRange(input, Scalar(min_hsv[0], 0, 0), Scalar(255, 255, 255), hi);
        inRange(input, Scalar(0, 0, 0), Scalar(max_hsv[0], 255, 255), lo);

        bitwise_or(hi, lo, temp);
        bitwise_and(temp, mask, output);
    }
    else {
        inRange(input, min_hsv, max_hsv, output);
    }
}

bool billiards::imgproc::is_border_pixel(cv::Rect img_size, cv::Vec2i pixel, int margin)
{
    pixel = pixel - (cv::Vec2i)img_size.tl();
    bool w = pixel[0] < margin || pixel[0] >= img_size.width - margin;
    bool h = pixel[1] < margin || pixel[1] >= img_size.height - margin;
    return w || h;
}
