#include "recognition_impl.hpp"
#include <iostream>
#include <numeric>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core/base.hpp>
#include <random>
#include <algorithm>

namespace billiards
{
void recognizer_impl_t::find_table(img_t const& img, recognition_desc& desc, const cv::Mat& rgb, const cv::UMat& filtered, vector<cv::Vec2f>& table_contours)
{
    using namespace cv;

    {
        vector<vector<Point>> candidates;
        vector<Vec4i> hierarchy;
        findContours(filtered, candidates, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        // drawContours(rgb, candidates, -1, {0, 0, 255}, 1);

        // 사각형 컨투어 찾기
        for (int idx = 0; idx < candidates.size(); ++idx) {
            auto& contour = candidates[idx];

            approxPolyDP(vector(contour), contour, m.table.polydp_approx_epsilon, true);
            auto area_size = contourArea(contour);
            if (area_size < m.table.min_pxl_area_threshold) {
                continue;
            }

            convexHull(vector(contour), contour, true);
            approxPolyDP(vector(contour), contour, m.table.polydp_approx_epsilon, true);
            // drawContours(rgb, candidates, -1, {255, 128, 0}, 3);
            putText(rgb, (stringstream() << "[" << contour.size() << ", " << area_size << "]").str(), contour[0], FONT_HERSHEY_PLAIN, 1.0, {0, 255, 0});

            bool const table_found = contour.size() == 4;

            if (table_found) {
                // drawContours(filtered, candidates, idx, {255}, 1);

                // marshal
                for (auto& pt : contour) {
                    table_contours.push_back(Vec2f(pt.x, pt.y));
                }
            }
        }
    }

    if (table_contours.size() == 4) {
        vector<Vec3f> obj_pts;
        Vec3d tvec;
        Vec3d rvec;

        {
            float half_x = m.table.recognition_size.val[0] / 2;
            float half_z = m.table.recognition_size.val[1] / 2;

            // ref: OpenCV coordinate system
            // 참고: findContour로 찾아진 외각 컨투어 세트는 항상 반시계방향으로 정렬됩니다.
            obj_pts.push_back({-half_x, 0, half_z});
            obj_pts.push_back({-half_x, 0, -half_z});
            obj_pts.push_back({half_x, 0, -half_z});
            obj_pts.push_back({half_x, 0, half_z});
        }

        // tvec의 초기값을 지정해주기 위해, 깊이 이미지를 이용하여 당구대 중점 위치를 추정합니다.
        bool estimation_valid = true;
        vector<cv::Vec3d> table_points_3d = {};
        {
            // 카메라 파라미터는 컬러 이미지 기준이므로, 깊이 이미지 해상도를 일치시킵니다.
            cv::Mat depth = img.depth;

            auto& c = img.camera;

            // 당구대의 네 점의 좌표를 적절한 3D 좌표로 변환합니다.
            for (auto& uv : table_contours) {
                auto u = uv[0];
                auto v = uv[1];
                auto z_metric = depth.at<float>(v, u);

                auto& pt = table_points_3d.emplace_back();
                pt[2] = z_metric;
                pt[0] = z_metric * ((u - c.cx) / c.fx);
                pt[1] = z_metric * ((v - c.cy) / c.fy);
            }
        }
        bool solve_successful = false;
        /*
        solve_successful = estimation_valid;
        /*/
        // 3D 테이블 포인트를 바탕으로 2D 포인트를 정렬합니다.
        // 모델 공간에서 테이블의 인덱스는 짧은 쿠션에서 시작해 긴 쿠션으로 반시계 방향 정렬된 상태입니다. 이미지에서 검출된 컨투어는 테이블의 반시계 방향 정렬만을 보장하므로, 모델 공간에서의 정점과 같은 순서가 되도록 contour를 재정렬합니다.
        {
            assert(table_contours.size() == table_points_3d.size());

            // 오차를 감안해 공간에서 변의 길이가 table size의 mean보다 작은 값을 선정합니다.
            auto thres = sum(m.table.recognition_size)[0] * 0.5;
            for (int idx = 0; idx < table_contours.size() - 1; idx++) {
                auto& t = table_points_3d;
                auto& c = table_contours;
                auto len = norm(t[idx + 1] - t[idx], NORM_L2);

                // 다음 인덱스까지의 거리가 문턱값보다 짧다면 해당 인덱스를 가장 앞으로 당깁니다(재정렬).
                if (len < thres) {
                    c.insert(c.end(), c.begin(), c.begin() + idx);
                    t.insert(t.end(), t.begin(), t.begin() + idx);
                    c.erase(c.begin(), c.begin() + idx);
                    t.erase(t.begin(), t.begin() + idx);

                    break;
                }
            }
            auto& p = img.camera;
            double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

            double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
            auto mat_cam = cv::Mat(3, 3, CV_64FC1, M);
            auto mat_disto = cv::Mat(4, 1, CV_64FC1, disto);

            solve_successful = solvePnP(obj_pts, table_contours, mat_cam, mat_disto, rvec, tvec, false, SOLVEPNP_ITERATIVE);

            //            auto error_estimate = norm(tvec_estimate - tvec);
        }
        //*/

        if (solve_successful) {
            Vec3f tvec_world = tvec, rvec_world = rvec;
            camera_to_world(img, rvec_world, tvec_world);

            vector<vector<Point>> contours;
            project_model(img, contours.emplace_back(), tvec_world, rvec_world, obj_pts, false);

            // 각 점을 비교하여 에러를 계산합니다.
            auto& proj = contours.front();
            int max_error = -1;
            for (size_t index = 0; index < 4; index++) {
                Vec2f projpt = (Vec2i)proj[index];
                max_error = max<int>(norm(projpt - table_contours[index], NORM_L1), max_error);
            }

            if (max_error < m.table.solvePnP_max_distance_error_threshold) {
                set_filtered_table_rot(rvec_world);
                set_filtered_table_pos(tvec_world);
                desc.table.confidence = 0.9f;
                draw_axes(img, (Mat&)rgb, rvec_world, tvec_world, 0.08f, 3);
                drawContours(rgb, contours, -1, {0, 255, 0}, 3);
            }
            else {
                drawContours(rgb, contours, -1, {0, 0, 0}, 1);
            }
        }

        desc.table.position = table_pos_flt;
        desc.table.orientation = (Vec4f&)table_rot_flt;
    }
}

cv::Vec3f recognizer_impl_t::set_filtered_table_pos(cv::Vec3f new_pos, float confidence)
{
    float alpha = m.table.LPF_alpha_pos * confidence;
    return table_pos_flt = (1 - alpha) * table_pos_flt + alpha * new_pos;
}

cv::Vec3f recognizer_impl_t::set_filtered_table_rot(cv::Vec3f new_rot, float confidence)
{
    if (norm(table_rot_flt - new_rot) > (170.0f) * CV_PI / 180.0f) {
        // rotation[1] += CV_PI;
        new_rot = rotate_local(new_rot, {0, (float)CV_PI, 0});
    }

    float alpha = m.table.LPF_alpha_rot * confidence;
    return table_rot_flt = (1 - alpha) * table_rot_flt + alpha * new_rot;
}

void recognizer_impl_t::cull_frustum(float hfov_rad, float vfov_rad, vector<cv::Vec3f>& obj_pts)
{
    using namespace cv;
    // 시야 사각뿔의 4개 평면은 반드시 원점을 지납니다.
    // 평면은 N.dot(P-P1)=0 꼴인데, 평면 상의 임의의 점 P1은 원점으로 설정해 노멀만으로 평면을 나타냅니다.
    vector<plane_t> planes;
    {
        // horizontal 평면 = zx 평면
        // vertical 평면 = yz 평면
        Matx33f rot_vfov;
        Rodrigues(Vec3f(vfov_rad * 0.5f, 0, 0), rot_vfov); // x축 양의 회전
        planes.push_back({rot_vfov * Vec3f{0, 1, 0}, 0});  // 위쪽 면

        Rodrigues(Vec3f(-vfov_rad * 0.5f, 0, 0), rot_vfov);
        planes.push_back({rot_vfov * Vec3f{0, -1, 0}, 0}); // 아래쪽 면

        Rodrigues(Vec3f(0, hfov_rad * 0.5f, 0), rot_vfov);
        planes.push_back({rot_vfov * Vec3f{-1, 0, 0}, 0}); // 오른쪽 면

        Rodrigues(Vec3f(0, -hfov_rad * 0.5f, 0), rot_vfov);
        planes.push_back({rot_vfov * Vec3f{1, 0, 0}, 0}); // 오른쪽 면
    }

    // 먼저, 절두체 내에 있는 점을 찾고 해당 점을 탐색의 시작점으로 지정합니다.
    bool cull_whole = true;
    for (size_t idx = 0; idx < obj_pts.size(); idx++) {
        bool is_inside = true;

        for (auto& plane : planes) {
            if (plane.calc(obj_pts[idx]) < 0.f) {
                is_inside = false;
                break;
            }
        }

        if (is_inside) {
            // 절두체 내부에 있는 점이 가장 앞에 오게끔 ...
            obj_pts.insert(obj_pts.end(), obj_pts.begin(), obj_pts.begin() + idx);
            obj_pts.erase(obj_pts.begin(), obj_pts.begin() + idx);
            cull_whole = false;
            break;
        }
    }

    // 평면 내에 있는 점이 하나도 없다면, 드랍합니다.
    if (cull_whole) {
        obj_pts.clear();
        return;
    }

    for (size_t idx = 0; idx < obj_pts.size(); idx++) {
        size_t nidx = idx + 1 >= obj_pts.size() ? 0 : idx + 1;
        auto& o = obj_pts;

        // 각 평면에 대해 ...
        for (auto& plane : planes) {
            // 이 때, idx는 반드시 평면 안에 있습니다.
            if (plane.calc(o[idx]) < 0) {
                continue;
            }

            if (plane.has_contact(o[idx], o[nidx]) == false) {
                // 이 문맥에서는 반드시 o[idx]가 평면 위에 위치합니다.
                continue;
            }

            // 평면 위->아래로 진입하는 문맥
            // 접점 상에 새 점을 삽입합니다.
            {
                auto contact = plane.find_contact(o[idx], o[nidx]).value();
                o.insert(o.begin() + nidx, contact);
            }

            // nidx부터 0번 인덱스까지, 다시 진입해 들어오는 직선을 찾습니다.
            for (size_t pivot = nidx + 1; pivot < o.size();) {
                auto next_idx = pivot + 1 >= o.size() ? 0 : pivot + 1;

                if (!plane.has_contact(o[pivot], o[next_idx])) {
                    // 만약 접점이 없다면, 현재 직선의 시작점은 완전히 평면 밖에 있으므로 삭제합니다.
                    // nidx2의 엘리먼트가 idx2로 밀려 들어오므로, 인덱스 증감은 따로 일어나지 않음
                    o.erase(o.begin() + pivot);
                }
                else {
                    // 접점이 존재한다면 현재 위치의 점을 접점으로 대체하고 break
                    auto contact = plane.find_contact(o[pivot], o[next_idx]).value();
                    o[pivot] = contact;
                    break;
                }
            }

            // 현재 인덱스를 건드리지 않으므로, 다른 평면은 딱히 변하지 않습니다.
            // 또한 새로 생성된 점은 반드시 기존 직선 위에서 생성되므로 평면의 순서는 관계 없습니다.
        }
    }
}

void recognizer_impl_t::get_world_transform_matx(cv::Vec3f pos, cv::Vec3f rot, cv::Mat& world_transform)
{
    world_transform = cv::Mat(4, 4, CV_32FC1);
    world_transform.setTo(0);
    world_transform.at<float>(3, 3) = 1.0f;
    {
        // Vec3f rot = (Vec3f&)desc.table.orientation;
        auto tr_mat = world_transform({0, 3}, {3, 4});
        auto rot_mat = world_transform({0, 3}, {0, 3});
        copyTo(pos, tr_mat, {});
        Rodrigues(rot, rot_mat);
    }
}

void recognizer_impl_t::get_camera_matx(img_t const& img, cv::Mat& mat_cam, cv::Mat& mat_disto)
{
    auto& p = img.camera;
    double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

    double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
    mat_cam = cv::Mat(3, 3, CV_64FC1, M).clone();
    mat_disto = cv::Mat(4, 1, CV_64FC1, disto).clone();
}

void recognizer_impl_t::get_table_model(std::vector<cv::Vec3f>& vertexes, cv::Vec2f model_size)
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

bool recognizer_impl_t::get_safe_ROI_rect(cv::Mat const& mat, cv::Rect& roi)
{
    if (roi.x >= mat.cols || roi.y >= mat.rows) {
        goto RETURN_FALSE;
    }

    if (roi.x < 0) {
        roi.width += roi.x;
        roi.x = 0;
    }
    if (roi.y < 0) {
        roi.height += roi.y;
        roi.y = 0;
    }

    if (roi.x + roi.width >= mat.cols) {
        roi.width -= (roi.x + roi.width + 1) - mat.cols;
    }

    if (roi.y + roi.height >= mat.rows) {
        roi.height -= (roi.y + roi.height + 1) - mat.rows;
    }

    if (roi.width <= 0 || roi.height <= 0) {
        goto RETURN_FALSE;
    }

    return true;

RETURN_FALSE:;
    roi.width = roi.height = 0;
    return false;
}

std::optional<cv::Mat> recognizer_impl_t::get_safe_ROI(cv::Mat const& mat, cv::Rect roi)
{
    using namespace cv;

    if (get_safe_ROI_rect(mat, roi)) {
        return mat(roi);
    }

    return {};
}

void recognizer_impl_t::get_point_coord_3d(img_t const& img, float& io_x, float& io_y, float z_metric)
{
    auto& c = img.camera;
    auto u = io_x;
    auto v = io_y;

    io_x = z_metric * ((u - c.cx) / c.fx);
    io_y = z_metric * ((v - c.cy) / c.fy);
}

array<float, 2> recognizer_impl_t::get_uv_from_3d(img_t const& img, cv::Point3f const& coord_3d)
{
    array<float, 2> result;
    auto& [u, v] = result;
    auto& [x, y, z] = coord_3d;
    auto c = img.camera;

    u = (c.fx * x) / z + c.cx;
    v = (c.fy * y) / z + c.cy;

    return result;
}

void recognizer_impl_t::filter_hsv(cv::InputArray input, cv::OutputArray output, cv::Vec3f min_hsv, cv::Vec3f max_hsv)
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

void recognizer_impl_t::camera_to_world(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec) const
{
    using namespace cv;
    vector<Vec3f> uvw;
    uvw.emplace_back(0.1f, 0, 0);
    uvw.emplace_back(0, -0.1f, 0);
    uvw.emplace_back(0, 0, 0.1f);
    uvw.emplace_back(0, 0, 0);

    Matx33f rot;
    Rodrigues(rvec, rot);

    for (auto& pt : uvw) {
        pt = (rot * pt) + tvec;

        auto pt4 = (Vec4f&)pt;
        pt4[3] = 1.0f, pt4[1] *= -1.0f;
        pt4 = img.camera_transform * pt4;
        pt = (Vec3f&)pt4;
    }

    auto u = normalize(uvw[0] - uvw[3]);
    auto v = normalize(uvw[1] - uvw[3]);
    auto w = normalize(uvw[2] - uvw[3]);
    tvec = uvw[3];

    Mat1f rotation(3, 3);
    copyTo(u, rotation.col(0), {});
    copyTo(v, rotation.col(1), {});
    copyTo(w, rotation.col(2), {});

    Rodrigues(rotation, rvec);

    // uvw가 제대로 프로젝트되었는지 확인
    // Mat dst = img.rgb.clone();
    // draw_circle(img, dst, 10.0f, tvec + u * 0.1f, {0, 0, 255});
    // draw_circle(img, dst, 10.0f, tvec + v * 0.1f, {0, 255, 0});
    // draw_circle(img, dst, 10.0f, tvec + w * 0.1f, {255, 0, 0});
    // draw_circle(img, dst, 10.0f, uvw[3], {255, 255, 255});
    // const_cast<recognizer_impl_t*>(this)->show("axes", dst);
}

cv::Vec3f recognizer_impl_t::rotate_local(cv::Vec3f target, cv::Vec3f rvec)
{
    using namespace cv;
    Matx33f axes = Matx33f::eye();
    Matx33f rotator;
    Rodrigues(target, rotator);
    axes = rotator * axes;

    auto roll = rvec[2] * axes.col(2);
    auto pitch = rvec[0] * axes.col(0);
    auto yaw = rvec[1] * axes.col(1);

    Rodrigues(roll, rotator), axes = rotator * axes;
    Rodrigues(pitch, rotator), axes = rotator * axes;
    Rodrigues(yaw, rotator), axes = rotator * axes;

    Vec3f result;
    Rodrigues(axes, result);
    return result;
}

void recognizer_impl_t::draw_axes(img_t const& img, cv::Mat& dest, cv::Vec3f rvec, cv::Vec3f tvec, float marker_length, int thickness) const
{
    using namespace cv;
    vector<Vec3f> pts;
    pts.assign({{0, 0, 0}, {marker_length, 0, 0}, {0, -marker_length, 0}, {0, 0, marker_length}});

    vector<Vec2f> mapped;
    project_model(img, mapped, tvec, rvec, pts, false);

    pair<int, int> pairs[] = {{0, 1}, {0, 2}, {0, 3}};
    Scalar colors[] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
    for (int i = 0; i < 3; ++i) {
        auto [beg, end] = pairs[i];
        auto color = colors[i];

        Point pt_beg(mapped[beg][0], mapped[beg][1]);
        Point pt_end(mapped[end][0], mapped[end][1]);
        line(dest, pt_beg, pt_end, color, thickness);
    }
}

void recognizer_impl_t::draw_circle(img_t const& img, cv::Mat& dest, float base_size, cv::Vec3f tvec_world, cv::Scalar color) const
{
    using namespace cv;
    vector<Vec3f> pos{{0, 0, 0}};
    vector<Vec2f> pt;

    project_model(img, pt, tvec_world, Vec3f(), pos, false);

    float size = base_size / norm(pos);
    circle(dest, Point(pt[0][0], pt[0][1]), size, color, -1);
}

plane_t plane_t::from_NP(cv::Vec3f N, cv::Vec3f P)
{
    N = cv::normalize(N);

    plane_t plane;
    plane.N = N;
    plane.d = 0.f;

    auto u = plane.calc_u(P, P + N).value();

    plane.d = -u;
    return plane;
}

plane_t& plane_t::transform(cv::Vec3f tvec, cv::Vec3f rvec)
{
    using namespace cv;

    auto P = N * d;
    Matx33f rotator;
    Rodrigues(rvec, rotator);

    N = rotator * N;
    P = rotator * P + tvec;

    return *this = from_NP(N, P);
}

float plane_t::calc(cv::Vec3f const& pt) const
{
    auto res = cv::sum(N.mul(pt))[0] + d;
    return abs(res) < 1e-6f ? 0 : res;
}

bool plane_t::has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    return calc(P1) * calc(P2) < 0.f;
}

optional<float> plane_t::calc_u(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    auto P3 = N * d;

    auto upper = N.dot(P3 - P1);
    auto lower = N.dot(P2 - P1);

    if (abs(lower) > 1e-6f) {
        auto u = upper / lower;

        return u;
    }

    return {};
}

optional<cv::Vec3f> plane_t::find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    if (auto uo = calc_u(P1, P2)) {
        auto u = *uo;

        if (u <= 1.f && u >= 0.f) {
            return P1 + (P2 - P1) * u;
        }
    }
    return {};
}

void recognizer_impl_t::transform_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, vector<cv::Vec3f>& model_vertexes)
{
    cv::Mat world_transform;
    get_world_transform_matx(world_pos, world_rot, world_transform);

    cv::Mat inv_camera_transform = cv::Mat(img.camera_transform).inv();
    for (auto& opt : model_vertexes) {
        auto pt = (cv::Vec4f&)opt;
        pt[3] = 1.0f;

        pt = *(cv::Vec4f*)cv::Mat(inv_camera_transform * world_transform * pt).data;

        // 좌표계 변환
        pt[1] *= -1.0f;
        opt = (cv::Vec3f&)pt;
    }
}

void recognizer_impl_t::project_model(img_t const& img, vector<cv::Vec2f>& mapped_contours, cv::Vec3f world_pos, cv::Vec3f world_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull)
{
    transform_to_camera(img, world_pos, world_rot, model_vertexes);

    // 오브젝트 포인트에 frustum culling 수행
    if (do_cull) {
        cull_frustum(90 * CV_PI / 180.0f, 60 * CV_PI / 180.0f, model_vertexes);
    }

    if (!model_vertexes.empty()) {
        // obj_pts 점을 카메라에 대한 상대 좌표로 치환합니다.
        cv::Mat mat_cam, mat_disto;
        get_camera_matx(img, mat_cam, mat_disto);

        // 각 점을 매핑합니다.
        projectPoints(model_vertexes, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), mat_cam, mat_disto, mapped_contours);
    }
}

void billiards::recognizer_impl_t::project_model(img_t const& img, vector<cv::Point>& mapped, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull)
{
    vector<cv::Vec2f> mapped_vec;
    project_model(img, mapped_vec, obj_pos, obj_rot, model_vertexes, do_cull);

    mapped.clear();
    for (auto& pt : mapped_vec) {
        mapped.emplace_back((int)pt[0], (int)pt[1]);
    }
}

void recognizer_impl_t::correct_table_pos(img_t const& img, recognition_desc& desc, cv::Mat rgb, cv::Rect ROI, cv::Mat3b roi_rgb, vector<cv::Point> table_contour_partial)
{
    cv::Mat mat_cam, mat_disto;
    get_camera_matx(img, mat_cam, mat_disto);
    auto aruco_dict = cv::aruco::getPredefinedDictionary(m.table.aruco_dictionary);
    auto aruco_param = cv::aruco::DetectorParameters::create();

    vector<int> all_marker;
    vector<vector<cv::Point2f>> all_corner;
    vector<float> marker_distances;

    struct contour_desc_t
    {
        cv::Point screen_point, offset_point;
        float distance;
    };

    vector<contour_desc_t> cached_location_contours;
    {
        vector<cv::Vec3f> obj_pts;
        get_table_model(obj_pts, m.table.recognition_size);

        // 정점을 하나씩 투사합니다.
        // frustum culling을 배제하기 위함입니다.
        vector<cv::Vec2f> points;
        project_model(img, points, table_pos_flt, table_rot_flt, obj_pts, false);

        { // debug rendering
            vector<vector<cv::Point>> draw_point(1);
            for (auto& pt : points) { draw_point.front().emplace_back(pt[0], pt[1]); }

            drawContours(rgb, draw_point, -1, {255, 255, 0}, 2);
        }

        // 오프셋을 계산합니다.
        vector<cv::Vec3f> marker_pts;
        vector<cv::Vec2f> marker_proj_pts;
        get_table_model(marker_pts, m.table.recognition_size);
        {
            auto [x, z] = m.table.aruco_offset_from_corner;
            marker_pts[0] += cv::Vec3f(-x, 0, z);
            marker_pts[1] += cv::Vec3f(-x, 0, -z);
            marker_pts[2] += cv::Vec3f(x, 0, -z);
            marker_pts[3] += cv::Vec3f(x, 0, z);
        }
        project_model(img, marker_proj_pts, table_pos_flt, table_rot_flt, marker_pts, false);

        for (int i = 0; i < obj_pts.size(); ++i) {
            auto& c = cached_location_contours.emplace_back();
            c.distance = norm(obj_pts[i]);
            c.screen_point = cv::Point(points[i][0], points[i][1]);
            c.offset_point = cv::Point(marker_proj_pts[i][0], marker_proj_pts[i][1]) - c.screen_point;
            c.screen_point -= ROI.tl();
        }
    }

    for (auto& pt : table_contour_partial) {
        circle(roi_rgb, pt, 5, {255, 64, 0}, -1);

        // 당구대 모서리의 예측 지점과의 화면상 거리를 계산합니다.
        float dist_scr_min = numeric_limits<float>::max();
        contour_desc_t* arg = {};
        for (auto& s : cached_location_contours) {
            float dist = norm(s.screen_point - pt);
            if (dist < dist_scr_min) {
                dist_scr_min = dist;
                arg = &s;
            }
        }

        if (dist_scr_min > m.table.pixel_distance_threshold_per_meter / arg->distance) {
            continue;
        }

        // 테이블 에지 주변을 ROI로 지정하고, ArUco 디텍션 수행
        // 거리 획득
        auto center = pt + arg->offset_point;
        int size = m.table.aruco_detection_rect_radius_per_meter / arg->distance;

        cv::Rect ROI_small(center - cv::Point(size, size), center + cv::Point(size, size));
        rectangle(roi_rgb, ROI_small, {255, 0, 255}, 2);

        cv::Mat small_roi;
        if (auto roi = get_safe_ROI(img.rgba(ROI), ROI_small)) {
            cvtColor(*roi, small_roi, cv::COLOR_RGBA2RGB);
        }
        else {
            continue;
        }

        // 전처리
        subtract(255, small_roi, small_roi);

        vector<int> marker_ids;
        vector<vector<cv::Point2f>> marker_corners, rejected_cands;
        detectMarkers(small_roi, aruco_dict, marker_corners, marker_ids, aruco_param, rejected_cands);

        for (auto& shapes : marker_corners) {
            for (auto& corner : shapes) {
                corner += cv::Point2f(ROI_small.tl()) + cv::Point2f(ROI.tl());
            }
        }

        all_marker.insert(all_marker.end(), marker_ids.begin(), marker_ids.end());
        all_corner.insert(all_corner.end(), marker_corners.begin(), marker_corners.end());

        for (int i = 0; i < marker_corners.size(); ++i) { marker_distances.push_back(arg->distance); }
    }

    cv::aruco::drawDetectedMarkers(rgb, all_corner, all_marker, {0, 0, 255});

    if (marker_distances.empty() == false) {
        vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(all_corner, m.table.aruco_marker_size, mat_cam, mat_disto, rvecs, tvecs);

        // 모든 마커에 대해 ...
        for (int index = 0; index < 1; ++index) {
            cv::Vec3f rvec = rvecs[index], tvec = tvecs[index];
            int marker_index = all_marker[index];
            rvec = rotate_local(rvec, {-(float)CV_PI / 2, 0, 0});
            camera_to_world(img, rvec, tvec);
            draw_axes(img, rgb, rvec, tvec, 0.1, 3);

            // 마커 번호에서 인덱스 획득
            int position_index = -1;
            if (auto it = find(std::begin(m.table.aruco_index_map), std::end(m.table.aruco_index_map), marker_index);
                it != std::end(m.table.aruco_index_map)) {
                position_index = it - std::begin(m.table.aruco_index_map);
            }

            if (position_index != -1) {
                // 테이블 중점에서 마커의 위치만큼을 빼 줍니다.
                auto pt = m.table.aruco_horizontal_points[position_index];
                cv::Vec3f min_error_candidate = {};

                // 테이블의 방향은 보장되어있지 않으므로, 테이블의 회전에 대해 오차가 가장 적은 후보를 선택합니다.

                for (int i = 0; i < 2; ++i) {
                    cv::Matx33f world_rot;
                    cv::Vec3f table_rotation = rotate_local(table_rot_flt, {0, (float)CV_PI * i, 0});

                    Rodrigues((cv::Vec3f)table_rotation, world_rot);
                    auto table_pos_estimate = tvec - world_rot * pt;
                    auto err_my = norm(table_pos_estimate - (cv::Vec3f)table_pos_flt);
                    auto err_min = norm(min_error_candidate - (cv::Vec3f)table_pos_flt);

                    if (err_my < err_min) {
                        min_error_candidate = table_pos_estimate;
                    }
                }

                desc.table.orientation = (cv::Vec4f&)table_rot_flt;
                desc.table.position = set_filtered_table_pos(min_error_candidate, 0.25f / marker_distances[index]);
                desc.table.confidence = 0.7f;
            }
        }
    }
}

void recognizer_impl_t::find_ball_center(img_t const& img, vector<cv::Point> const& contours_src, ball_find_parameter_t const& p, ball_find_result_t& r)
{
    /*
    필요: 평면 카메라 좌표계 기준으로 축 변환하기(N, P 각각 카메라 트랜스폼으로 역변환)
    
    과정
        1. 컨투어 무게중심 측정
        2. 임의의 중점 후보 2D 좌표 선택
            해당 2D좌표를 깊은 Z값에 대한 임의의 카메라 좌표로 바꾸고, 카메라 기준 당구대 평면에 대해 투영하여 실제 컨택트 획득, Z 값으로부터 당구공의 픽셀 반경 계산.
        3. 적합성 검사 수행
            0. 임의 개수의 컨투어를 선정합니다(최적화).
            1. 픽셀 중점과 반경을 활용해 각 컨투어와 원의 경계선 사이의 유사도를 계산하고, 가까울수록 높은 가중치를 부여합니다. 컨투어는 불완전한 반원 형태로부터 추출되므로 이상치를 벗어난 점들은 큰 의미가 없기 때문에 RMSE는 사용하지 않습니다.
            |  같은 맥락에서, 에러가 가장 적어지는 점이 아닌 가중치가 가장 큰 점을 선택하게 됩니다.
        4. 2에서 선택한 중점 후보 중 가장 가중치가 높은 점을 선택하고, 반경을 반으로 줄여 다시 중점 후보를 선택합니다.
        5. 2~4의 과정을 정해진 횟수만큼 반복합니다.
    */
    using namespace cv;

    struct candidate_t
    {
        Vec2f uv = {};
        float radius = 0;
        float geometric_weight = 0;
        float color_weight = 0;
        float z = 0;
    };

    auto const& search = m.ball.search;

    Point2f search_center;
    float search_radius, max_weight = 0.f;
    minEnclosingCircle(contours_src, search_center, search_radius);
    search_radius *= search.candidate_radius_amp;

    uniform_real_distribution<float> distr_angle(0, CV_2PI);
    vector<Point> contours;

    // 컨투어 리스트를 최적화합니다.
    if (contours_src.size() > search.num_max_contours) {
        mt19937 rd_gen(clock());
        vector<int> indexes(contours_src.size());
        iota(indexes.begin(), indexes.end(), 0);
        shuffle(indexes.begin(), indexes.end(), rd_gen);

        indexes.erase(indexes.begin() + search.num_max_contours, indexes.end());
        sort(indexes.begin(), indexes.end());

        contours.reserve(search.num_max_contours);
        for (auto idx : indexes) { contours.emplace_back(contours_src[idx]); }
    }
    else {
        contours = contours_src;
    }

    // Lazy convolution을 수행합니다.
    unordered_map<uint64_t, float> color_weight_table__;
    Mat color_weight_kernel__;
    {
        int kernel_size = int(1.5f * search_radius / search.candidate_radius_amp) | 1; // 홀수로 만듭니다.
        int radius = (kernel_size - 1) / 2;
        color_weight_kernel__ = Mat(kernel_size, kernel_size, CV_32FC3, {0, 0, 0});
        //  circle(color_weight_kernel__, {radius, radius}, radius, p.hsv_avg_filter_value, -1);
        color_weight_kernel__.setTo(p.hsv_avg_filter_value / 255.0);
    }

    // 색상 적합도 계산을 위한 컨볼루션의 lazy evaluation을 수행하는 펑터입니다.
    // 메모이제이션 기법을 활용해, L2거리 일정 이내의 점에 대한 재계산을 생략합니다.
    auto color_weight = [&kernel = color_weight_kernel__,
                         &table = color_weight_table__,
                         &search,
                         &p](Point at, float dist) {
        // 포인트를 잘라내 해상도를 낮춥니다.
        // 낮아진 해상도는 반올림해 매핑됩니다.
        if (int opt = p.memoization_steps; opt > 1) {
            int adder = opt / 2;
            at.x = at.x + adder - at.x % opt;
            at.y = at.y + adder - at.y % opt;
        }
        uint64_t hash = ((0ull + at.x) << 32) + at.y;

        auto found_it = table.find(hash);
        if (found_it == table.end()) {
            // 색상 적합도 계산을 위해 해당 포인트를 중점으로 둔 ROI를 구합니다.
            Rect window_rect;
            window_rect.x = at.x - kernel.cols / 2;
            window_rect.y = at.y - kernel.rows / 2;
            window_rect.width = kernel.cols;
            window_rect.height = kernel.rows;
            window_rect += p.ROI.tl(); // 원본 이미지에 윈도우 생성

            if (!get_safe_ROI_rect(p.precomputed_color_weights, window_rect)) {
                return 0.0f;
            }

            float weight_sum = sum(p.precomputed_color_weights(window_rect))[0];

            // 가중치에서, 테이블의 파란 영역과 겹치는 만큼 가중치에 패널티를 줍니다.
            float penalty = sum(p.blue_mask(window_rect))[0] / 255;

            found_it = table.try_emplace(hash, weight_sum - penalty).first;
        }

        return found_it->second * dist * dist;
    };

    vector<candidate_t> candidates;
    candidate_t search_cand = {};
    search_cand.uv = search_center;

    for (int iteration = search.iterations; iteration--;) {
        // 다음 중점 후보 목록 생성
        uniform_real_distribution<float> distr_radius(0, search_radius);
        candidates.clear();

        // MULTITHREADING BEGIN
        for (int candidate_index = 0; candidate_index < search.num_candidates; ++candidate_index) {
            candidate_t cand;

            // 중점 후보의 UV 좌표 생성
            // 이 때, 첫 번째 요소는 이전 요소를 계승합니다.
            if (candidate_index == 0) {
                cand = search_cand;
            }
            else {
                auto seed = candidate_index + search.num_candidates * (iteration + 1);
                mt19937 rd_gen(seed);
                float angle = distr_angle(rd_gen);
                float radius = distr_radius(rd_gen);
                // float angle = candidate_index * ((float)CV_2PI) / search.num_candidates;
                // float radius = search_radius;

                Point2f ofst;
                ofst.x = radius * cosf(angle);
                ofst.y = radius * sinf(angle);

                cand.uv = search_center + ofst;
            }

            // 해당 UV 좌표를 당구대 평면으로 투사, 실제 픽셀 반경 계산
            Vec3f contact(cand.uv[0], cand.uv[1], 10.0f); // 10.0meter는 충분히 길음
            contact[0] += p.ROI.x;
            contact[1] += p.ROI.y;
            get_point_coord_3d(img, contact[0], contact[1], contact[2]);
            if (auto contact_opt = p.table_plane->find_contact({}, contact)) {
                contact = *contact_opt;
            }
            else {
                // 시야 위치에 따라 접점이 평면상에 아예 존재하지 않을 수 있습니다.
                continue;
            }

            // 새로 계산된 거리에 따라 해당 후보의 적합한 반경을 계산합니다.
            cand.radius = get_pixel_length(img, m.ball.radius, contact[2]);

            // 각 컨투어 목록을 방문하여 가중치를 계산합니다.
            // pow(base, -distance)를 이용해, 거리가 0이면 가장 높은 가중치, 멀어질수록 지수적으로 감소합니다.
            float weight = 0.f;
            float base = search.weight_function_base;
            for (auto& pt : contours) {
                Vec2f ptf(pt.x, pt.y);
                weight += ::pow(base, -abs(cand.radius - norm(ptf - cand.uv, NORM_L2)));
            }

            cand.geometric_weight = weight;
            cand.color_weight = color_weight((Point2f)cand.uv, contact[2]);
            cand.z = contact[2];

            candidates.emplace_back(cand);
        }
        // MULTITHREADING END

        if (candidates.empty()) {
            break;
        }

        auto& closest = *max_element(candidates.begin(), candidates.end(), [&search](candidate_t const& a, candidate_t const& b) { return (a.geometric_weight - b.geometric_weight) + (a.color_weight - b.color_weight) * search.color_weight < 0; });

        // Debug line 그리기
        Point2f ofst = p.ROI.tl();
        if (false && search.render_debug) {
            for (auto& cand : candidates) {
                line(p.rgb_debug, ofst + search_center, ofst + (Point2f)cand.uv, {0, 128, 0});
            }
        }

        if (closest.geometric_weight > search_cand.geometric_weight) {
            if (search.render_debug) {
                line(p.rgb_debug, ofst + search_center, ofst + Point2f(closest.uv), {255, 255, 255}, 2);
                circle(p.rgb_debug, ofst + Point2f(closest.uv), closest.radius, Scalar{0, 0, 128}, 1);
            }

            // 결과가 개선됐을 때만 해당 지점으로 이동합니다.
            r.img_center = search_center = closest.uv;
            r.img_center += p.ROI.tl();
            r.pixel_radius = closest.radius;
            r.geometric_weight = closest.geometric_weight;
            r.color_weight = closest.color_weight;

            (Vec2f&)r.ball_position = (Point2f)closest.uv + (Point2f)p.ROI.tl();
            r.ball_position.z = closest.z;
            get_point_coord_3d(img, r.ball_position.x, r.ball_position.y, r.ball_position.z);

            search_cand = closest;
        }
    }
}

void recognizer_impl_t::async_worker_thread()
{
    while (worker_is_alive) {
        {
            unique_lock<mutex> lck(worker_event_wait_mtx);
            worker_event_wait.wait(lck);
        }

        opt_img_t img;
        img_cb_t on_finish;
        if (read_lock lck(img_cue_mtx); img_cue.has_value()) {
            img = move(*img_cue);
            on_finish = move(img_cue_cb);
            img_cue = {};
            img_cue_cb = {};
        }

        if (img.has_value()) {
            img_show_queue.clear();
            auto desc = proc_img(*img);
            {
                write_lock lock(img_show_mtx);
                img_show = img_show_queue;
            }
            if (on_finish) { on_finish(*img, desc); }
            prev_desc = desc;
        }
    }
}

recognition_desc recognizer_impl_t::proc_img(img_t const& img)
{
    using namespace cv;
    recognition_desc desc;

    if (statics.empty()) {
    }

    vars = {};

    auto& p = m.props;

    // RGBA 이미지를 RGB로 컨버트합니다.
    Mat img_rgb;
    cvtColor(img.rgba, img_rgb, COLOR_RGBA2RGB);
    vars["img-rgb"] = img_rgb;

    // 공용 머터리얼 셋업 시퀀스
    Mat img_rgb_scaled, img_hsv_scaled;
    {
        // 스케일된 이미지 준비
        Size size_scaled_image((int)p["fast-process-width"], 0);
        float scale = size_scaled_image.width / (float)img_rgb.cols;
        size_scaled_image.height = (int)(img_rgb.rows * scale);

        resize(img_rgb, img_rgb_scaled, size_scaled_image);
        vars["img-rgb-scaled"] = img_rgb_scaled;

        // 스케일된 이미지의 파라미터 준비
        auto scp = img.camera;
        for (auto& value : {&scp.fx, &scp.fy, &scp.cx, &scp.cy}) {
            *value *= scale;
        }
        vars["camera-param-scaled"] = scp;

        // HSV 이미지 준비
        cvtColor(img_rgb_scaled, img_hsv_scaled, COLOR_RGB2HSV);
    }

    // 테이블 탐색을 위한 로직입니다.
    {
      // -- 테이블 색상으로 필터링 수행

      // -- 경계선 검출

      // -- 테이블 추정 영역 컨투어 찾기

      // -- CASE 1. 테이블 전체 시야에 들어오는 경우
      // - solvePnP 활용해서 테이블 포즈 찾기
      // - 오차에 따라 컨피던스 설정
      // - 값 LPF 필터 누적

      // -- CASE 2. 테이블 일부만 시야에 들어온 경우

      // - 오차 확인
      // - 임의의 각도로 재투영, 정해진 횟수만큼 iterate
    }

    // 공 탐색을 위한 로직입니다.
    {
        // -- 테이블 영역을 Perspective에서 Orthogonal하게 투영합니다.
        // (방법은 아직 연구가 필요 ..)
        // - 이미지는 이미 rectify된 상태이므로, 카메라 파라미터는 따로 고려하지 않음
        // - 테이블의 perspective point로부터 당구대 이미지 획득
        // - 해당 이미지를 테이블을 orthogonal로 투영한 이미지 영역으로 트랜스폼
        // (참고로, Orthogonal하게 투영된 이미지는 원근 X)

        // -- 각 색상의 가장 우수한 후보를 선정
        // - 위에서 Orthogonal Transform을 통해 얻은 이미지 사용
        // - 필드에서 빨강, 오렌지, 흰색 각 색상의 HSV 값을 뺌
        // - 뺀 값 각각에 가중치를 주어 합산(reduce) (H가 가장 크게, V를 가장 작게)
        // - 해당 값의 음수를 pow의 지수로 둠 ... pow(base, -weight); 즉 거리가 멀수록 0에 가까운 값 반환
        // - 고정 커널 크기(Orthogonal로 Transform했으므로 ..)로 컨볼루션 적용, 로컬 맥시멈 추출 ... 공 candidate


        
    }

    // ShowImage에 모든 임시 매트릭스 추가
    for (auto& pair : vars) {
        auto& value = pair.second;

        if (auto ptr = any_cast<Mat>(&value)) {
            show(move(pair.first), *ptr);
        }
        else if (auto ptr = any_cast<UMat>(&value)) {
            show(move(pair.first), *ptr);
        }
    }

    return desc;
} // namespace billiards

void recognizer_impl_t::plane_to_camera(img_t const& img, plane_t const& table_plane, plane_t& table_plane_camera)
{
    cv::Vec4f N = (cv::Vec4f&)table_plane.N;
    cv::Vec4f P = table_plane.d * N;
    N[3] = 0.f, P[3] = 1.f;

    cv::Matx44f camera_inv = img.camera_transform.inv();
    N = camera_inv * N;
    P = camera_inv * P;

    // CV 좌표계로 변환
    N[1] *= -1.f, P[1] *= -1.f;

    table_plane_camera = plane_t::from_NP((cv::Vec3f&)N, (cv::Vec3f&)P);
}

float recognizer_impl_t::get_pixel_length(img_t const& img, float len_metric, float Z_metric)
{
    using namespace cv;

    auto [u1, v1] = get_uv_from_3d(img, Vec3f(0, 0, Z_metric));
    auto [u2, v2] = get_uv_from_3d(img, Vec3f(len_metric, 0, Z_metric));

    return u2 - u1;
}

recognition_desc recognizer_impl_t::proc_img2(img_t const& img)
{
    using namespace cv;

    recognition_desc desc = {};

    TickMeter tick_tot;
    tick_tot.start();
    Mat hsv_all;
    Mat rgb_all_debug = img.rgba.clone();
    resize(img.depth, (Mat&)img.depth, {rgb_all_debug.cols, rgb_all_debug.rows});

    UMat table_blue_mask_gpu, table_blue_edges;
    {
        UMat ucolor;
        UMat b;
        rgb_all_debug.copyTo(ucolor);

        // 색공간 변환
        cvtColor(ucolor, b, COLOR_RGBA2RGB);
        b.copyTo(rgb_all_debug);
        cvtColor(b, ucolor, COLOR_RGB2HSV);
        ucolor.copyTo(hsv_all);
        // show("hsv", uclor);

        // 색역 필터링 및 에지 검출
        if (m.table.hsv_filter_max[0] < m.table.hsv_filter_min[0]) {
            UMat hi, lo;
            auto filt_min = m.table.hsv_filter_min, filt_max = m.table.hsv_filter_max;
            filt_min[0] = 0, filt_max[0] = 255;

            inRange(ucolor, filt_min, filt_max, table_blue_mask_gpu);
            inRange(ucolor, Scalar(m.table.hsv_filter_min[0], 0, 0), Scalar(255, 255, 255), hi);
            inRange(ucolor, Scalar(0, 0, 0), Scalar(m.table.hsv_filter_max[0], 255, 255), lo);

            bitwise_or(hi, lo, table_blue_edges);
            bitwise_and(table_blue_mask_gpu, table_blue_edges, table_blue_mask_gpu);

            // show("mask", mask);
        }
        else {
            inRange(ucolor, m.table.hsv_filter_min, m.table.hsv_filter_max, table_blue_mask_gpu);
        }
    }

    {
        UMat umat_temp;
        table_blue_mask_gpu.copyTo(umat_temp);

        GaussianBlur(umat_temp, umat_temp, {3, 3}, 5);
        threshold(umat_temp, table_blue_mask_gpu, 128, 255, THRESH_BINARY);

        erode(table_blue_mask_gpu, umat_temp, {}, {-1, -1}, 1);
        bitwise_xor(table_blue_mask_gpu, umat_temp, table_blue_edges);
    }

    // 테이블 위치 탐색
    vector<Vec2f> table_contours; // 2D 컨투어도 반환받습니다.
    find_table(img, desc, rgb_all_debug, table_blue_edges, table_contours);
    show("filtered", table_blue_edges);

    auto mm = Mat(img.camera_transform);

    /* 당구공 위치 찾기 */
    // 1. 당구대 ROI를 추출합니다.
    // 2. 당구대 중점(desc.table... or filtered...) 및 노멀(항상 {0, 1, 0})을 찾습니다.
    // 3. 당구공의 uv 중점 좌표를 이용해 당구공과 카메라를 잇는 직선을 찾습니다.
    // 4. 위 직선을 투사하여, 당구대 평면과 충돌하는 지점을 찾습니다.
    // 5. PROFIT!

    // 만약 contour_table이 비어 있다면, 이는 2D 이미지 내에 당구대 일부만 들어와있거나, 당구대가 없어 정확한 위치가 검출되지 않은 경우입니다. 따라서 먼저 당구대의 알려진 월드 위치를 transformation하여 화면 상에 투사한 뒤, contour_table을 구성해주어야 합니다.

    bool const table_not_detect = table_contours.empty();
    {
        Vec3f pos = table_pos_flt;
        Vec3f rot = table_rot_flt;

        vector<cv::Vec3f> obj_pts;
        get_table_model(obj_pts, m.table.outer_masking_size);

        project_model(img, table_contours, pos, rot, obj_pts);

        // debug draw contours
        if (!table_contours.empty()) {
            vector<vector<Point>> contour_draw;
            auto& tbl = contour_draw.emplace_back();
            for (auto& pt : table_contours) { tbl.push_back({(int)pt[0], (int)pt[1]}); }
            drawContours(rgb_all_debug, contour_draw, -1, {0, 0, 255}, 8);
        }
    }

    // ROI 추출
    Rect ROI;
    if (table_contours.empty() == false) {
        int xbeg, ybeg, xend, yend;
        xbeg = ybeg = numeric_limits<int>::max();
        xend = yend = -1;

        for (auto& pt : table_contours) {
            xbeg = min<int>(pt[0], xbeg);
            xend = max<int>(pt[0], xend);
            ybeg = min<int>(pt[1], ybeg);
            yend = max<int>(pt[1], yend);
        }

        xbeg = max(0, xbeg);
        ybeg = max(0, ybeg);
        xend = max(min(rgb_all_debug.cols - 1, xend), xbeg);
        yend = max(min(rgb_all_debug.rows - 1, yend), ybeg);
        ROI = Rect(xbeg, ybeg, xend - xbeg, yend - ybeg);
    }

    bool roi_valid = ROI.size().area() > 1000;

    // ROI 존재 = 당구 테이블이 시야 내에 있음
    if (roi_valid) {
        Mat3b roi_rgb;
        cvtColor(img.rgba(ROI), roi_rgb, COLOR_RGBA2RGB);
        UMat roi_edge_gpu;
        table_blue_mask_gpu(ROI).copyTo(roi_edge_gpu);
        Mat roi_mask(ROI.size(), roi_edge_gpu.type());
        roi_mask.setTo(0);

        // 현재 당구대 추정 위치 영역으로 마스크를 설정합니다.
        // 이미지 기반으로 하는 것보다 정확성은 다소 떨어지지만, 이미 당구대가 시야에서 벗어나 위치 추정이 어긋나기 시작하는 시점에서 정확성을 따질 겨를이 없습니다.
        {
            vector<vector<Point>> contours_;
            auto& contour = contours_.emplace_back();
            for (auto& pt : table_contours) { contour.emplace_back((int)pt[0] - ROI.x, (int)pt[1] - ROI.y); }

            // 컨투어 영역만큼의 마스크를 그립니다.
            // drawContours(roi_mask, contours_, 0, {255}, FILLED);
        }

        // ROI 내에서, 당구대 영역을 재지정합니다.
        {
            UMat sub;

            // 당구대 영역으로 마스킹 수행
            // roi_edge = roi_edge.mul(roi_mask);

            // 에지 검출 이전에, 팽창-침식 연산을 통해 에지를 단순화하고, 파편을 줄입니다.
            GaussianBlur(roi_edge_gpu, roi_edge_gpu, {3, 3}, 15);
            threshold(roi_edge_gpu, roi_edge_gpu, 128, 255, THRESH_BINARY);

            // 내부에 닫힌 도형을 만들수 있게끔, 경계선을 깎아냅니다.
            roi_edge_gpu.row(0).setTo(0);
            roi_edge_gpu.row(roi_edge_gpu.rows - 1).setTo(0);
            roi_edge_gpu.col(0).setTo(0);
            roi_edge_gpu.col(roi_edge_gpu.cols - 1).setTo(0);

            erode(roi_edge_gpu, sub, {});
            bitwise_xor(roi_edge_gpu, sub, roi_edge_gpu);
            show("edge_new", roi_edge_gpu);
        }

        vector<Point> table_contour_partial;
        {
            vector<vector<Point>> candidates;
            vector<Vec4i> hierarchy;
            findContours(roi_edge_gpu, candidates, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            int max_size_index = -1;
            for (int idx = 0, max_area = -1; idx < hierarchy.size(); ++idx) {
                if (candidates[idx].size() > 3) {
                    auto area = contourArea(candidates[idx]);
                    if (area > max_area) {
                        max_area = area;
                        max_size_index = idx;
                    }
                }
            }

            // 테이블 컨투어를 찾습니다.
            if (candidates.empty() == false && max_size_index >= 0) {
                table_contour_partial = candidates[max_size_index];
            }
        }

        if (false && table_not_detect && table_contour_partial.empty() == false) {
            approxPolyDP(table_contour_partial, table_contour_partial, 25, true);

            correct_table_pos(img, desc, rgb_all_debug, ROI, roi_rgb, table_contour_partial);
        }

        show("partial-view", roi_rgb);

        Rect ROI_fit = {};
        {
            // 테이블 컨투어를 재조정합니다.
            vector<Vec3f> obj_pts;
            get_table_model(obj_pts, m.table.inner_size);
            project_model(img, table_contour_partial, table_pos_flt, table_rot_flt, obj_pts, true);
            ROI_fit = boundingRect(table_contour_partial);
            get_safe_ROI_rect(img.rgba, ROI_fit);

            if (table_contour_partial.empty() == false) {
                drawContours(rgb_all_debug, vector({table_contour_partial}), -1, {255, 255, 255}, 2);
            }
        }

        // 당구공을 찾기 위해 ROI를 더 알맞게 재조정합니다.
        if (ROI_fit.area() > 0) {
            {
                for (auto& pt : table_contour_partial) { pt -= ROI_fit.tl(); }

                roi_mask = Mat(ROI_fit.size(), CV_8U).setTo(0);
                drawContours(roi_mask, vector<vector<Point>>{table_contour_partial}, -1, {255}, -1);
            }

            // 테이블 마스크를 빼줍니다. 테이블 외의 부분만 추출해내기 위함입니다.
            Mat roi_area_mask = roi_mask.clone();
            Mat roi_area_mask_invert = 255 - roi_area_mask;
            Mat roi_table_excluded_rgb;
            subtract(roi_mask, table_blue_mask_gpu(ROI_fit), roi_mask);

            {
                Mat temp;
                cvtColor(img.rgba(ROI_fit), temp, COLOR_RGBA2RGB);
                roi_rgb.release();
                temp.copyTo(roi_rgb, roi_mask);
                roi_rgb.copyTo(roi_table_excluded_rgb);
            }

            // 마스크로부터 에지를 구합니다.
            Mat edge;
            table_blue_edges(ROI_fit).copyTo(edge);
            edge.setTo(0, roi_area_mask_invert);

            // 컨투어 리스트 획득, 각각에 대해 반복합니다.
            vector<vector<Point>> non_table_contours;
            findContours(edge, non_table_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            drawContours(roi_rgb, non_table_contours, -1, {0, 255, 0}, 1);

            // 월드 당구대 평면을 획득합니다.
            plane_t table_plane, table_plane_camera;
            {
                auto P = table_pos_flt;
                Matx33f rotator;
                Rodrigues(table_rot_flt, rotator);
                auto N = rotator * Vec3f{0, 1, 0};
                table_plane = plane_t::from_NP(N, P);
            }

            // 카메라 기준으로 변환
            plane_to_camera(img, table_plane, table_plane_camera);

            int ball_index = 0;
            Mat table_blue_mask = table_blue_mask_gpu.getMat(ACCESS_FAST).clone();

            ball_find_parameter_t ball_param;
            Vec3f* ball_color_filters[3] = {m.ball.color.red, m.ball.color.orange, m.ball.color.white};
            Mat ball_param_precomputed_mat[3];
            {
                // 파라미터를 셋업
                ball_param.table_plane = &table_plane_camera;
                ball_param.rgb_debug = rgb_all_debug;
                ball_param.blue_mask = table_blue_mask;

                Scalar base = log(m.ball.search.color_kernel_weight_base);
                UMat A, B, uhsv;
                hsv_all.copyTo(B);
                B.convertTo(uhsv, CV_32FC3, 1.0 / 255);

                // TODO: uhsv 로 바꾸기,
                for (int index = 0; index < 3; ++index) {
                    auto filt = ball_color_filters[index];
                    auto color = (filt[0] + filt[1]) * (0.5f * (1.0f / 255));
                    Mat mat;

                    subtract(uhsv, (Scalar)color, B);
                    multiply(B, B, A);
                    multiply(A, Scalar(2.0, 1.0, 0.0), B);
                    cv::reduce(B.reshape(1, B.rows * B.cols), A, 1, REDUCE_SUM);
                    B = A.reshape(1, B.rows);

                    // Element-wise pow 계산 a^b = e^bln(a)
                    sqrt(B, A);
                    multiply(A, -base * m.ball.search.color_dist_amplitude, B);
                    exp(B, A);

                    A.copyTo(mat);
                    ball_param_precomputed_mat[index] = mat;
                }
            }

            vector<ball_find_result_t> ball_results[3];
            for (auto& ball_chunk_contours : non_table_contours) {
                // auto mm = moments(ball_chunk_contours);
                // auto size = mm.m00;
                // int cent_x = ROI_fit.x + mm.m10 / size;
                // int cent_y = ROI_fit.y + mm.m01 / size;
                Point2f contour_circle_center;
                float contour_circle_radius;
                minEnclosingCircle(ball_chunk_contours, contour_circle_center, contour_circle_radius);
                auto [cent_x, cent_y] = contour_circle_center + (Point2f)ROI_fit.tl();
                auto size = contour_circle_radius;
                size = size * size * CV_PI;

                circle(rgb_all_debug, (Point)contour_circle_center + ROI_fit.tl(), contour_circle_radius, {0, 0, 255});

                // 카메라와 당구공 사이의 거리를 구합니다.
                float dist_between_cam = 10000000.f;
                {
                    Vec3f P2(cent_x, cent_y, 5.f);
                    get_point_coord_3d(img, P2[0], P2[1], P2[2]);

                    if (auto ball_pos = table_plane_camera.find_contact({}, P2)) {
                        dist_between_cam = ball_pos->val[2]; // Z 값
                    }
                }

                float eval = size * dist_between_cam * dist_between_cam;
                if (eval < m.ball.pixel_count_per_meter_min || eval > m.ball.pixel_count_per_meter_max) {
                    continue;
                }

                float pixel_radius = get_pixel_length(img, m.ball.radius, dist_between_cam);
                // circle(rgb_all_debug, {cent_x, cent_y}, pixel_radius, {0, 255, 0}, 2, LINE_8);

                // 당구공 영역의 ROI 추출합니다.
                auto ROI_ball = boundingRect(ball_chunk_contours) + ROI_fit.tl();

                // 거리에 따라 해상도를 동적으로 조절합니다.
                ball_param.ROI = ROI_ball;
                ball_param.memoization_steps = pixel_radius * m.ball.search.memoization_distance_rate;

                // 모든 ball 색상에 대해 필터링 수행
                Mat color = hsv_all(ROI_ball), filtered, edge;
                for (auto& pt : ball_chunk_contours) { pt -= ROI_ball.tl() - ROI_fit.tl(); }

                // 각각의 색상에 대한 에지를 구하고, 합성합니다.
                for (int index = 0; index < 3; ++index) {
                    auto filter = ball_color_filters[index];

                    // 색상으로 필터링 및 에지 검출
                    {
                        filter_hsv(color, filtered, filter[0], filter[1]);
                        filtered.setTo(0, roi_area_mask_invert(ROI_ball - ROI_fit.tl()));

                        // 가우시안 필터링을 통해 노이즈를 제거하고, 침식을 통해 에지 검출
                        GaussianBlur(filtered, filtered, {5, 5}, 150.0);
                        threshold(filtered, filtered, 128, 255, THRESH_BINARY);
                        erode(filtered, edge, {}, {-1, -1}, 1);
                        filtered.row(0).setTo(0);
                        filtered.row(filtered.rows - 1).setTo(0);
                        filtered.col(0).setTo(0);
                        filtered.col(filtered.cols - 1).setTo(0);
                        bitwise_xor(filtered, edge, edge);
                    }

                    // 이렇게 계산된 에지로부터 컨투어를 추출합니다.
                    vector<vector<Point>> shapes;
                    findContours(edge, shapes, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                    if (m.ball.search.render_debug) {
                        // drawContours(rgb_all_debug(ROI_ball), shapes, -1, {0, 0, 0}, -1);
                        drawContours(rgb_all_debug(ROI_ball), shapes, -1, {64, 128, 63}, 1);
                        //   if (index == 0)
                        //       show((stringstream() << "ball " << ball_index << " filter " << index).str(), edge);
                    }
                    ball_param.hsv_avg_filter_value = (filter[0] + filter[1]) * 0.5f;
                    ball_param.precomputed_color_weights = ball_param_precomputed_mat[index];

                    // 각 컨투어 후보를 반복하여 공의 중심을 찾습니다.
                    // TODO: 각 색상별 공 개수 및 컨투어 영역 크기를 활용해 invalid한 candidate 걸러내기
                    // TODO: 빨간색의 경우, 검출 후 검출 영역을 잘라낸 영역 재검토, 공 두 개 겹친 경우 걸러내기 위함. (영역 안에 없는 컨투어만 모아서 배열 만들기)
                    for (auto& contours : shapes) {
                        // 지금까지 찾아낸 모든 원 후보를 반복합니다.

                        // TODO: Area size가 공 하나의 크기보다 크면, 같은 색상의 공이 두 개 이상 겹쳐 있을 가능성 고려
                        if (contourArea(contours) < pixel_radius * pixel_radius) {
                            continue;
                        }

                        // 각 색상 별 탐색 결과를 모두 수집한 뒤, 웨이트가 가장 높고 이전 결과와 연관성이 높은 후보를 공으로 선정합니다.
                        auto& res = ball_results[index].emplace_back();
                        find_ball_center(img, contours, ball_param, res);

                        if (res.geometric_weight < 2.f) {
                            ball_results[index].pop_back();
                            continue;
                        }

                        // 디버그 그리기
                        {
                            auto center = res.img_center;
                            Mat3b ball_color_rgb(1, 1);
                            ball_color_rgb.setTo((Scalar)ball_param.hsv_avg_filter_value);
                            cvtColor(ball_color_rgb, ball_color_rgb, COLOR_HSV2RGB);
                            circle(rgb_all_debug, center, res.pixel_radius, ball_color_rgb(0), 1);
                            putText(rgb_all_debug, to_string(res.geometric_weight), center + Point(0, -7), FONT_HERSHEY_PLAIN, 1, {0, 255, 0});
                            putText(rgb_all_debug, to_string(res.color_weight), center + Point(0, 7), FONT_HERSHEY_PLAIN, 1, {0, 255, 255});
                        }
                    }
                }

                // 공 검출 결과를 선별해 유니티로 보냅니다.
                {
                    auto predicate = [&m = this->m](ball_find_result_t const& a, ball_find_result_t const& b) { return a.geometric_weight - b.geometric_weight + (a.color_weight - b.color_weight) * m.ball.search.color_weight > 0; };

                    for (auto& results : ball_results) {
                        // 엘리먼트를 내림차순 정렬합니다.
                        sort(results.begin(), results.end(), predicate);
                    }

                    auto pivot = m.ball.search.confidence_pivot_weight;
                    recognition_desc::ball_recognition_result* order[] = {&desc.ball.red1, &desc.ball.red2, &desc.ball.orange, &desc.ball.white};
                    int table_indexes[] = {0, 0, 1, 2};
                    int elem_indexes[] = {0, 1, 0, 0};

                    for (int index = 0; index < 4; ++index) {
                        auto& table = ball_results[table_indexes[index]];
                        auto& dest = *order[index];
                        int elem_idx = elem_indexes[index];

                        if (table.size() >= elem_idx + 1) {
                            auto& elem = table[elem_idx];
                            dest.confidence = (elem.color_weight + elem.geometric_weight) / pivot;
                            Vec3f position = elem.ball_position;
                            Vec3f placeholder_ = {};
                            camera_to_world(img, placeholder_, position);
                            memcpy(dest.position, position.val, sizeof dest.position);
                        }
                        else {
                            dest.confidence = 0.f;
                        }
                    }
                }

                ++ball_index;
            }
            show("fit_mask", roi_rgb);
            show("fit_edge", edge);
        }
    }

    // 결과물 출력
    tick_tot.stop();
    float elapsed = tick_tot.getTimeMilli();
    putText(rgb_all_debug, (stringstream() << "Elapsed: " << elapsed << " ms").str(), {0, rgb_all_debug.rows - 5}, FONT_HERSHEY_PLAIN, 1.0, {255, 255, 255});
    show("source", rgb_all_debug);
    return desc;
}

recognizer_t::recognizer_t()
    : impl_(make_unique<recognizer_impl_t>(*this))
{
}

recognizer_t::~recognizer_t() = default;

void recognizer_t::refresh_image(parameter_type image, recognizer_t::process_finish_callback_type&& callback)
{
    auto& m = *impl_;
    bool img_swap_before_prev_img_proc = false;

    if (write_lock lock(m.img_cue_mtx); lock) {
        img_swap_before_prev_img_proc = !!m.img_cue;
        m.img_cue = image;
        m.img_cue_cb = move(callback);
    }

    if (img_swap_before_prev_img_proc) {
        cout << "warning: image request cued before previous image processed\n";
    }

    m.worker_event_wait.notify_all();
}

void recognizer_t::poll(std::unordered_map<std::string, cv::Mat>& shows)
{
    // 비동기적으로 수집된 이미지 목록을 획득합니다.
    auto& m = *impl_;

    if (read_lock lock(m.img_show_mtx, try_to_lock); lock) {
        for (auto& pair : m.img_show) {
            shows[pair.first] = pair.second;
        }
    }
}
} // namespace billiards
