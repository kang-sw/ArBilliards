#include "recognizer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

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
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    auto& table_contour = *i.table_contours;
    auto& img = *i.img_ptr;

    bool is_any_border_point = [&]() {
        for (auto pt : table_contour) {
            if (is_border_pixel({{}, i.img_size}, pt)) return true;
        }
        return false;
    }();

    if (table_contour.size() == 4 && !is_any_border_point) {
        PIPEPP_ELAPSE_SCOPE("PNP Solver");

        vector<Vec3f> obj_pts;
        Vec3d tvec, rvec;
        float max_confidence = 0;

        get_table_model(obj_pts, i.table_fit_size);
        auto [mat_cam, mat_disto] = get_camera_matx(img);

        // 테이블의 방향을 고려하여, 그대로의 인덱스와 시프트한 인덱스 각각에 대해 PnP 알고리즘을 적용, 포즈를 계산합니다.
        for (int i = 0; i < 2; ++i) {
            Vec3d pos, rot;
            if (!solvePnP(obj_pts, table_contour, mat_cam, mat_disto, rot, pos)) {
                continue;
            }

            // confidence 계산
            auto vertexes = obj_pts;
            for (auto& vtx : vertexes) {
                vtx = rodrigues(rvec) * vtx + tvec;
            }

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
        }
    }
    return {};
}

void billiards::imgproc::cull_frustum_impl(std::vector<cv::Vec3f>& obj_pts, plane_t const* plane_ptr, size_t num_planes)
{
    using namespace cv;
    // 시야 사각뿔의 4개 평면은 반드시 원점을 지납니다.
    // 평면은 N.dot(P-P1)=0 꼴인데, 평면 상의 임의의 점 P1은 원점으로 설정해 노멀만으로 평면을 나타냅니다.
    auto planes = std::initializer_list<plane_t>(plane_ptr, plane_ptr + num_planes);
    assert(obj_pts.size() >= 3);

    for (auto pl : planes) {
        auto& o = obj_pts;
        constexpr auto SMALL_NUMBER = 1e-5f;
        // 평면 안의 점 찾기 ... 시작점
        int idx = -1;
        for (int i = 0; i < o.size(); ++i) {
            if (pl.calc(o[i]) >= SMALL_NUMBER) {
                idx = i;
                break;
            }
        }

        // 평면 안에 점 하나도 없으면 드랍
        if (idx == -1) {
            o.clear();
            return;
        }

        for (int nidx, incount = 0; incount < o.size();) {
            nidx = (idx + 1) % o.size();

            // o[idx]는 항상 평면 안

            // o[nidx]도 평면 안에 있다면 스킵
            if (pl.calc(o[nidx]) >= -SMALL_NUMBER) {
                ++incount;
                idx = ++idx % o.size();
                continue;
            }
            incount = 0; // 한 바퀴 더 돌아서 탈출하게 됨

            // o[idx]가 평면 안에 있는 것으로 가정하지만, 애매하게 걸치는 경우 발생
            // 이 경우 o[idx]를 삽입한 점으로 보고, o[nidx]에서 다음 프로세스 시작

            // o[idx]-o[nidx]는 평면을 위에서 아래로 통과
            // 접점 위치에 새로운 점 스폰(o[nidx] 위치)
            // nidx := nidx+1
            if (pl.calc(o[idx]) > 0) {
                auto contact = pl.find_contact(o[idx], o[nidx]).value();
                o.insert(o.begin() + nidx, contact);
                nidx = ++nidx % o.size();
            }

            // o[nidx]에서 출발, 다시 평면 위로 돌아올 때까지 반복
            // A. o[nidx]~o[nidx+1]이 모두 평면 밖에 있다면 o[nidx]는 제거
            // B. 다시 평면 위로 돌아온다면, 평면에 접점 스폰하고 o[nidx]를 대체
            for (int nnidx;;) {
                nnidx = (nidx + 1) % o.size();

                // o[nidx]는 반드시 평면 밖에 있음!
                // o[nnidx]도 평면 밖인 경우 ...
                if (pl.calc(o[nnidx]) < SMALL_NUMBER) {
                    o.erase(o.begin() + nidx);
                    nidx = nidx % o.size(); // index validate
                    continue;
                }

                // 단, o[nidx]가 평면 상에 있는 점일 수 있음
                // 이 경우 이미 o[nidx]를 스폰한 것으로 보고 다음 과정 진행
                if (pl.calc(o[nidx]) < 0) {
                    // o[nnidx]는 평면 안에 있음
                    // 접점을 스폰하고 nidx폐기, 탈출
                    auto contact = pl.find_contact(o[nidx], o[nnidx]).value();
                    o[nidx] = contact;
                }

                idx = nnidx; // nidx는 검증 완료, nnidx에서 새로 시작
                break;
            }
        }
    }
}

void billiards::imgproc::cull_frustum(std::vector<cv::Vec3f>& obj_pts, std::vector<plane_t> const& planes)
{
    cull_frustum_impl(obj_pts, planes.data(), planes.size());
}

void billiards::imgproc::project_model_local(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes)
{
    // 오브젝트 포인트에 frustum culling 수행
    if (do_cull) {
        cull_frustum(model_vertexes, planes);
    }

    if (!model_vertexes.empty()) {
        // obj_pts 점을 카메라에 대한 상대 좌표로 치환합니다.
        auto [mat_cam, mat_disto] = get_camera_matx(img);

        // 각 점을 매핑합니다.
        // projectPoints(model_vertexes, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), mat_cam, mat_disto, mapped_contour);
        project_points(model_vertexes, mat_cam, mat_disto, mapped_contour);
    }
}

void billiards::imgproc::project_points(std::vector<cv::Vec3f> const& points, cv::Matx33f const& camera, cv::Matx41f const& disto, std::vector<cv::Vec2f>& o_points)
{
    for (auto& pt : points) {
        auto intm = camera * pt;
        intm /= intm[2];
        o_points.emplace_back(intm[0], intm[1]);
    }
}

billiards::imgproc::plane_t billiards::imgproc::plane_t::from_NP(cv::Vec3f N, cv::Vec3f P)
{
    N = cv::normalize(N);

    plane_t plane;
    plane.N = N;
    plane.d = 0.f;

    // auto u = plane.calc_u(P, P + N).value();
    auto d = plane.calc(P);

    plane.d = -d;
    return plane;
}

billiards::imgproc::plane_t billiards::imgproc::plane_t::from_rp(cv::Vec3f rvec, cv::Vec3f tvec, cv::Vec3f up)
{
    using namespace cv;
    auto P = tvec;
    Matx33f rotator = rodrigues(rvec);
    // Matx33f rotator;
    // Rodrigues(rvec, rotator);
    auto N = rotator * up;
    return plane_t::from_NP(N, P);
}

billiards::imgproc::plane_t& billiards::imgproc::plane_t::transform(cv::Vec3f tvec, cv::Vec3f rvec)
{
    using namespace cv;

    auto P = -N * d;
    Matx33f rotator;
    Rodrigues(rvec, rotator);

    N = rotator * N;
    P = rotator * P + tvec;

    return *this = from_NP(N, P);
}

float billiards::imgproc::plane_t::calc(cv::Vec3f const& pt) const
{
    auto v = N.mul(pt);
    auto res = v[0] + v[1] + v[2] + d;
    return res; //abs(res) < 1e-6f ? 0 : res;
}

bool billiards::imgproc::plane_t::has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    return !!find_contact(P1, P2);
}

std::optional<float> billiards::imgproc::plane_t::calc_u(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    auto P3 = -N * d;

    auto upper = N.dot(P3 - P1);
    auto lower = N.dot(P2 - P1);

    if (abs(lower) > 1e-7f) {
        auto u = upper / lower;

        return u;
    }

    return {};
}

std::optional<cv::Vec3f> billiards::imgproc::plane_t::find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    if (auto uo = calc_u(P1, P2); uo /*&& calc(P1) * calc(P2) < 0*/) {
        auto u = *uo;

        if (u <= 1.f && u >= 0.f) {
            return P1 + (P2 - P1) * u;
        }
    }
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

void billiards::imgproc::get_table_model(std::vector<cv::Vec3f>& vertexes, cv::Vec2f model_size)
{
    vertexes.clear();
    auto [half_x, half_z] = (model_size * 0.5f).val;
    vertexes.assign(
      {
        {-half_x, 0, half_z},
        {-half_x, 0, -half_z},
        {half_x, 0, -half_z},
        {half_x, 0, half_z},
      });
}

std::pair<cv::Matx33d, cv::Matx41d> billiards::imgproc::get_camera_matx(billiards::recognizer_t::parameter_type const& img)
{
    auto& p = img.camera;
    double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

    double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
    return {cv::Matx33d(M), cv::Matx41d(disto)};
}
