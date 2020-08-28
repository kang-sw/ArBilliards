#include "recognition.hpp"
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <condition_variable>
#include <iostream>
#include <map>
#include <set>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace billiards
{
using img_t = recognizer_t::parameter_type;
using img_cb_t = recognizer_t::process_finish_callback_type;
using opt_img_t = optional<img_t>;
using read_lock = shared_lock<shared_mutex>;
using write_lock = unique_lock<shared_mutex>;
class recognizer_impl_t
{
public:
    recognizer_t& m;

    thread worker;
    atomic_bool worker_is_alive;
    condition_variable_any worker_event_wait;
    mutex worker_event_wait_mtx;

    opt_img_t img_cue;
    shared_mutex img_cue_mtx;
    img_cb_t img_cue_cb;

    map<string, cv::Mat> img_show;
    shared_mutex img_show_mtx;

    recognition_desc prev_desc;
    cv::Vec3d table_pos_flt = {};
    cv::Vec3d table_rot_flt = {};
    double table_yaw_flt = 0;

public:
    recognizer_impl_t(recognizer_t& owner)
        : m(owner)
    {
        worker_is_alive = true;
        worker = thread(&recognizer_impl_t::async_worker_thread, this);
    }

    ~recognizer_impl_t()
    {
        if (worker.joinable()) {
            worker_is_alive = false;
            worker_event_wait.notify_all();
            worker.join();
        }
    }

    void show(string wnd_name, cv::UMat img)
    {
        show(move(wnd_name), img.getMat(cv::ACCESS_FAST).clone());
    }

    void show(string wnd_name, cv::Mat img)
    {
        write_lock lock(img_show_mtx);
        img_show[move(wnd_name)] = move(img);
    }

    void async_worker_thread()
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
                auto desc = proc_img(*img);
                if (on_finish) { on_finish(*img, desc); }
                prev_desc = desc;
            }
        }
    }

    recognition_desc proc_img(img_t const& img);
    void find_table(img_t const& img, recognition_desc& desc, const cv::Mat& rgb, const cv::UMat& filtered, vector<cv::Vec2f>& table_contours);

    static void frustum_culling(float hfov_rad, float vfov_rad, vector<cv::Vec3f>& obj_pts);
};

void recognizer_impl_t::find_table(img_t const& img, recognition_desc& desc, const cv::Mat& rgb, const cv::UMat& filtered, vector<cv::Vec2f>& table_contours)
{
    using namespace cv;

    {
        vector<vector<Point>> candidates;
        vector<Vec4i> hierarchy;
        findContours(filtered, candidates, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        // 사각형 컨투어 찾기
        for (int idx = 0; idx < candidates.size(); ++idx) {
            auto& contour = candidates[idx];
            approxPolyDP(vector(contour), contour, m.table.polydp_approx_epsilon, true);

            auto area_size = contourArea(contour);
            bool const table_found = area_size > m.table.min_pxl_area_threshold && contour.size() == 4;

            // drawContours(rgb, candidates, idx, {0, 0, 255});

            if (table_found) {
                drawContours(rgb, candidates, idx, {255, 255, 255}, 2);

                // marshal
                for (auto& pt : contour) {
                    table_contours.push_back(Vec2f(pt.x, pt.y));
                }

                putText(rgb, (stringstream() << "[" << contour.size() << ", " << area_size << "]").str(), contour[0], FONT_HERSHEY_PLAIN, 1.0, {0, 255, 0});
            }
        }
    }

    if (table_contours.size() == 4) {
        vector<Vec3f> obj_pts;
        Mat tvec;
        Mat rvec;

        {
            float half_x = m.table.size.val[0] / 2;
            float half_z = m.table.size.val[1] / 2;

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
            cv::Mat depth;
            resize(img.depth, depth, {img.rgb.cols, img.rgb.rows});

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

            // tvec은 평균치를 사용합니다.
            {
                cv::Vec3d tvec_init = {};
                for (auto& pt : table_points_3d) {
                    tvec_init += pt;
                }
                tvec_init /= static_cast<float>(table_points_3d.size());
                tvec = cv::Mat(tvec_init);
                //  rvec = tvec;
            }

            // 테이블 포인트의 면적을 계산합니다.
            // 만약 정규 값보다 10% 이상 오차가 발생하면, 드랍합니다.
            {
                auto const& t = table_points_3d;
                auto sz_desired = m.table.size[0] * m.table.size[1];

                auto sz1 = 0.5 * norm((t[2] - t[1]).cross(t[0] - t[1]));
                auto sz2 = 0.5 * norm((t[0] - t[3]).cross(t[2] - t[3]));
                auto size = sz1 + sz2;

                auto err = abs(sz_desired - size);

                if (err > sz_desired * 0.3) {
                    estimation_valid = false;
                }
            }
        }

        bool solve_successful = false;
        bool use_pnp_data = false;
        /*
        solve_successful = estimation_valid;
        /*/
        // 3D 테이블 포인트를 바탕으로 2D 포인트를 정렬합니다.
        // 모델 공간에서 테이블의 인덱스는 짧은 쿠션에서 시작해 긴 쿠션으로 반시계 방향 정렬된 상태입니다. 이미지에서 검출된 컨투어는 테이블의 반시계 방향 정렬만을 보장하므로, 모델 공간에서의 정점과 같은 순서가 되도록 contour를 재정렬합니다.
        if (estimation_valid) {
            assert(table_contours.size() == table_points_3d.size());

            // 오차를 감안해 공간에서 변의 길이가 table size의 mean보다 작은 값을 선정합니다.
            auto thres = sum(m.table.size)[0] * 0.5;
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

            auto tvec_estimate = tvec.clone();
            solve_successful = solvePnP(obj_pts, table_contours, mat_cam, mat_disto, rvec, tvec, false, SOLVEPNP_ITERATIVE);

            auto error_estimate = norm(tvec_estimate - tvec);

            if (error_estimate > 0.2) {
                tvec = tvec_estimate;
            }
            else {
                use_pnp_data = true;
            }
        }
        //*/

        if (solve_successful) {
            {
                cv::Vec2f sum = {};
                for (auto v : table_contours) { sum += v; }
                cv::Point draw_at = {int(sum.val[0] / 4), int(sum.val[1] / 4)};

                auto translation = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                double dist = sqrt(translation.dot(translation));
                int thickness = max<int>(1, 2.0 / dist);

                putText(rgb, (stringstream() << dist).str(), draw_at, cv::FONT_HERSHEY_PLAIN, 2.0 / dist, {0, 0, 255}, thickness);
            }

            // tvec은 카메라 기준의 상대 좌표를 담고 있습니다.
            // img 구조체 인스턴스에는 카메라의 월드 트랜스폼이 담겨 있습니다.
            // tvec을 카메라의 orientation만큼 회전시키고, 여기에 카메라의 translation을 적용하면 물체의 월드 좌표를 알 수 있습니다.
            // rvec에는 카메라의 orientation을 곱해줍니다.
            {
                cv::Vec4d pos = cv::Vec4d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2), 0);
                pos[1] = -pos[1], pos[3] = 1.0;
                pos = img.camera_transform * pos;

                auto alpha = m.table.LPF_alpha_pos;
                table_pos_flt = table_pos_flt * (1 - alpha) + (cv::Vec3d&)pos * alpha;

                desc.table.position = table_pos_flt;
            }

            // 테이블의 각 점에 월드 트랜스폼을 적용합니다.
            for (auto& pt : table_points_3d) {
                cv::Vec4d pos = (cv::Vec4d&)pt;
                pos[3] = 1.0, pos[1] *= -1.0;

                pos = img.camera_transform * (cv::Vec4f)pos;
                pt = (cv::Vec3d&)pos;
            }

            // 테이블의 Yaw 방향 회전을 계산합니다.
            if (false) {
                // 모델 방향은 항상 우측(x축) 방향 고정입니다
                cv::Vec3d table_model_dir(1, 0, 0);

                // 먼저 테이블의 긴 방향 벡터를 구해야 합니다.
                // 테이블의 대각선 방향 벡터 두 개의 합으로 계산합니다.
                cv::Vec3d table_world_dir;
                {
                    auto& t = table_points_3d;
                    auto v1 = t[2] - t[0];
                    auto v2 = t[3] - t[1];

                    // 테이블의 월드 정점의 순서에 따라 v2, v1의 벡터 순서가 뒤바뀔 수 있습니다. 이 경우 두 벡터의 합은 짧은 방향을 가리키게 됩니다. 이를 방지하기 위해, 두 벡터 사이의 각도가 둔각이라면(즉, 짧은 방향을 가리킨다면) v2를 반전합니다.
                    auto theta = acos(v1.dot(v2) / (norm(v1) * norm(v2)));

                    if (theta >= CV_PI / 2)
                        v2 = -v2;

                    // 수직 방향 값을 suppress하고, 노멀을 구합니다.
                    // 당구대가 항상 수평 상태라고 가정하고, 약간의 오차를 무시합니다.
                    table_world_dir = normalize((v1 + v2).mul({1, 0, 1}));
                }

                // 각도를 계산하고, 외적을 통해 각도의 방향 또한 계산합니다.
                auto yaw_rad = acos(table_model_dir.dot(table_world_dir)); // unit vector
                yaw_rad *= table_world_dir.cross(table_model_dir)[1] > 0 ? 1.0 : -1.0;

                auto alpha = m.table.LPF_alpha_rot;
                table_yaw_flt = table_yaw_flt * (1 - alpha) + yaw_rad * alpha;
                desc.table.orientation = Vec4f(0, -table_yaw_flt, 0, 0);
            }

            // 테이블의 Yaw 방향 회전을 계산합니다.
            // 에러를 최소화하는 방향으로 회전 반복 적용합니다.
            {
                auto& p = img.camera;
                float disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

                float M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
                auto mat_cam = cv::Mat(3, 3, CV_32FC1, M);
                auto mat_disto = cv::Mat(4, 1, CV_32FC1, disto);

                Mat inv_camera_transform = Mat(img.camera_transform).inv();
                Mat world_transform(4, 4, CV_32FC1);
                world_transform.setTo(0);
                world_transform.at<float>(3, 3) = 1.0f;
                {
                    Vec3f pos = table_pos_flt;
                    // Vec3f rot = (Vec3f&)desc.table.orientation;
                    auto tr_mat = world_transform({0, 3}, {3, 4});
                    copyTo(pos, tr_mat, {});
                }

                vector<Vec4f> pt_origin;
                vector<Vec3f> proj_pts;
                for (auto& pt : obj_pts) { pt_origin.emplace_back((Vec4f&)pt)[3] = 1.0f; }

                // 오차를 구하는 함수입니다.
                auto f = [&](float x, bool draw = false) {
                    // 회전 각도만큼 rodrigues 생성, 월드 트랜스폼에 반영
                    Vec3f rot(0, -x, 0);
                    Rodrigues(rot, world_transform({0, 3}, {0, 3}));

                    // 월드 트랜스폼 적용 + 카메라 공간으로 트랜스폼
                    proj_pts.clear();
                    for (auto& pt : pt_origin) {
                        Mat tr_pt = (inv_camera_transform * world_transform * pt);
                        proj_pts.emplace_back(*(Vec3f*)tr_pt.data)[1] *= -1.0f;
                    }

                    // 화면에 투영합니다.
                    vector<Vec2f> proj_contours;
                    projectPoints(proj_pts, Vec3f{0, 0, 0}, Vec3f{0, 0, 0}, mat_cam, mat_disto, proj_contours);

                    if (draw) {
                        vector<vector<Point>> ptda;
                        ptda.emplace_back();
                        for (auto& pt : proj_contours) {
                            ptda[0].push_back({int(pt[0]), int(pt[1])});
                        }
                        drawContours(rgb, ptda, 0, {0, 255, 255}, 2);
                    }

                    // 오차를 계산합니다. 오차는 모든 점의 거리 합입니다.
                    // 이 때 점 순서는 고려하지 않고 가장 작은 오차 선정
                    assert(proj_contours.size() == table_contours.size());
                    float err_sum_min = numeric_limits<float>::max();
                    for (size_t idx_offset = 0; idx_offset < proj_contours.size(); ++idx_offset) {
                        float err_sum = 0;
                        for (size_t idx = 0; idx < proj_contours.size(); ++idx) {
                            size_t idx_n = (idx + idx_offset) % proj_contours.size();
                            Vec2f vec = proj_contours[idx] - table_contours[idx_n];
                            err_sum += norm(vec);
                        }

                        err_sum_min = min(err_sum, err_sum_min);
                    }

                    return err_sum_min;
                };

                // 180개의 값 찾기
                vector<float> all;
                float x2 = numeric_limits<float>::max();
                int at = -1;
                int divide = 90;
                for (int i = 0; i < divide; ++i) {
                    float f_i = f(i * (CV_PI / divide), false);
                    all.push_back(f_i);
                    if (f_i < x2) {
                        at = i;
                        x2 = f_i;
                    }
                }

                x2 = at * CV_PI / divide;
                if (isnan(x2)) { x2 = 0.0f; }

                // 요 적용
                auto alpha = m.table.LPF_alpha_rot;
                table_yaw_flt = table_yaw_flt * (1 - alpha) + x2 * alpha;
                desc.table.orientation = Vec4f(0, -table_yaw_flt, 0, 0);
            }

            // rvec의 validity를 테스트합니다.
            if (true) {
                auto& p = img.camera;
                double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

                double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
                auto mat_cam = cv::Mat(3, 3, CV_64FC1, M);
                auto mat_disto = cv::Mat(4, 1, CV_64FC1, disto);
                auto test_pts = obj_pts;
                test_pts.push_back({0.08, 0, 0});
                test_pts.push_back({0, 0.08, 0});
                test_pts.push_back({0, 0, 0.08});
                test_pts.push_back({0, 0, 0});
                Matx33d rot;
                Rodrigues(rvec, rot);

                {
                    for (auto& pt : test_pts) {
                        pt = (rot * (Vec3d)pt) + *(Vec3d*)tvec.data;
                    }
                    vector<Vec2f> proj_src;
                    projectPoints(test_pts, Vec3f::zeros(), Vec3f::zeros(), mat_cam, mat_disto, proj_src);

                    vector<Point> proj;
                    for (auto& pt : proj_src) { proj.push_back({(int)pt[0], (int)pt[1]}); }

                    vector<vector<Point>> contour_draw;
                    auto& tbl = contour_draw.emplace_back();

                    int iter = 4;
                    for (auto& pt : proj) {
                        if (iter-- > 0) tbl.push_back(pt);
                    }

                    drawContours(rgb, contour_draw, 0, {0, 255, 0}, 3);

                    line(rgb, proj[7], proj[4], {0, 0, 255}, 3);
                    line(rgb, proj[7], proj[5], {0, 255, 0}, 3);
                    line(rgb, proj[7], proj[6], {255, 0, 0}, 3);
                }
            }

            if (true) {
                // 전체 트랜스폼 계산 후, 회전만 추출
                // [u v w; Q] 행렬,
                Mat1d coord = Mat1d::eye(4, 4);
                coord({3, 4}, {0, 3}).setTo(1.0);

                Mat1d transform
                  = Mat1d::eye(4, 4);
                Rodrigues(rvec, transform({0, 0, 3, 3}));
                copyTo(tvec, transform({0, 3}, {3, 4}), {});

                Mat1d camera = Mat1d((Matx44d)img.camera_transform);
                coord = (camera * transform * coord).t();

                Vec3d rotation;
                Rodrigues(coord({0, 0, 3, 3}), rotation);

                table_rot_flt = rotation;
            }

            if (false) {
                // solvePnP로 검출된 로테이션 데이터를 사용하는 케이스입니다.
                // 먼저 Rodrigues 표현법으로 저장된 벡터를 회전행렬로 바꾸고, 이를 카메라 트랜스폼으로 전환합니다.
                Mat rot_mat(4, 4, CV_32FC1);
                rot_mat.setTo(0);
                rot_mat.at<float>(3, 3) = 1.0f;
                rvec.convertTo(rvec, CV_32FC1);
                // Rodrigues(rvec, rot_mat({0, 3}, {0, 3}));

                // Rodrigues(rot_mat({0, 3}, {0, 3}), rvec);
                // rvec.push_back(0.f);
                // rvec = img.camera_transform * rvec;

                desc.table.orientation = *(Vec4f*)rvec.data;
                desc.table.orientation[1] *= -1;
            }

            desc.table.orientation = (Vec4d&)table_rot_flt;
            desc.table.orientation[1] *= -1.0;
            desc.table.confidence = 0.9f;
        }
    }
}

// 평면을 나타내는 타입입니다.
struct plane_type
{
    cv::Vec3f N;
    float d;

    float calc(cv::Vec3f const& pt) const
    {
        auto res = cv::sum(N.mul(pt))[0] + d;
        return abs(res) < 1e-6f ? 0 : res;
    }

    bool has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
    {
        return calc(P1) * calc(P2) < 0.f;
    }

    optional<cv::Vec3f> find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
    {
        auto P3 = N * d;

        auto upper = N.dot(P3 - P1);
        auto lower = N.dot(P2 - P1);

        if (abs(lower) > 1e-6f) {
            auto u = upper / lower;

            if (u >= 0.f && u <= 1.f) {
                return P1 + u * (P2 - P1);
            }
        }

        return {};
    }
};

void recognizer_impl_t::frustum_culling(float hfov_rad, float vfov_rad, vector<cv::Vec3f>& obj_pts)
{
    using namespace cv;
    // 시야 사각뿔의 4개 평면은 반드시 원점을 지납니다.
    // 평면은 N.dot(P-P1)=0 꼴인데, 평면 상의 임의의 점 P1은 원점으로 설정해 노멀만으로 평면을 나타냅니다.
    vector<plane_type> planes;
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
            assert(plane.calc(o[idx]) >= 0);

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

recognition_desc recognizer_impl_t::proc_img(img_t const& img)
{
    using namespace cv;

    recognition_desc desc = {};

    TickMeter tick_tot;
    tick_tot.start();
    auto& rgb = img.rgb.clone();

    UMat uclor;
    Mat hsv;
    rgb.copyTo(uclor);

    // 색공간 변환
    cvtColor(uclor, uclor, COLOR_RGBA2RGB);
    cvtColor(uclor, uclor, COLOR_RGB2HSV);
    uclor.copyTo(hsv);
    show("hsv", uclor);

    // 색역 필터링 및 에지 검출
    UMat mask, filtered;
    if (m.table.hue_filter_max < m.table.hue_filter_min) {
        UMat hi, lo;
        inRange(uclor, m.table.sv_filter_min, m.table.sv_filter_max, mask);
        inRange(uclor, Scalar(m.table.hue_filter_min, 0, 0), Scalar(255, 255, 255), hi);
        inRange(uclor, Scalar(0, 0, 0), Scalar(m.table.hue_filter_max, 255, 255), lo);

        bitwise_or(hi, lo, filtered);
        bitwise_and(mask, filtered, mask);

        show("mask", mask);
    }
    else {
        inRange(uclor, m.table.sv_filter_min, m.table.sv_filter_max, mask);
    }

    {
        UMat umat_temp;
        // dilate(mask, umat_temp, {}, {-1, -1}, 4);
        // erode(umat_temp, mask, {}, {-1, -1}, 4);
        dilate(mask, umat_temp, {}, {-1, -1}, 1);
        bitwise_xor(mask, umat_temp, filtered);
    }
    show("filtered", filtered);

    // 테이블 위치 탐색
    vector<Vec2f> table_contours; // 2D 컨투어도 반환받습니다.
    find_table(img, desc, rgb, filtered, table_contours);
    auto mm = Mat(img.camera_transform);

    /* 당구공 위치 찾기 */
    // 1. 당구대 ROI를 추출합니다.
    // 2. 당구대 중점(desc.table... or filtered...) 및 노멀(항상 {0, 1, 0})을 찾습니다.
    // 3. 당구공의 uv 중점 좌표를 이용해 당구공과 카메라를 잇는 직선을 찾습니다.
    // 4. 위 직선을 투사하여, 당구대 평면과 충돌하는 지점을 찾습니다.
    // 5. PROFIT!

    // 만약 contour_table이 비어 있다면, 이는 2D 이미지 내에 당구대 일부만 들어와있거나, 당구대가 없어 정확한 위치가 검출되지 않은 경우입니다. 따라서 먼저 당구대의 알려진 월드 위치를 transformation하여 화면 상에 투사한 뒤, contour_table을 구성해주어야 합니다.
    {
        auto& p = img.camera;
        double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

        double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
        auto mat_cam = cv::Mat(3, 3, CV_64FC1, M);
        auto mat_disto = cv::Mat(4, 1, CV_64FC1, disto);

        vector<cv::Vec3f> obj_pts;
        {
            float half_x = m.table.size[0] / 2;
            float half_z = m.table.size[1] / 2;

            obj_pts.push_back({-half_x, 0, half_z});
            obj_pts.push_back({-half_x, 0, -half_z});
            obj_pts.push_back({half_x, 0, -half_z});
            obj_pts.push_back({half_x, 0, half_z});
        }

        // obj_pts 점을 카메라에 대한 상대 좌표로 치환합니다.
        Mat inv_camera_transform = Mat(img.camera_transform).inv();
        Mat world_transform(4, 4, CV_32FC1);
        world_transform.setTo(0);
        world_transform.at<float>(3, 3) = 1.0f;
        {
            Vec3f pos = table_pos_flt;
            Vec3f rot = table_rot_flt;
            // Vec3f rot = (Vec3f&)desc.table.orientation;
            auto tr_mat = world_transform({0, 3}, {3, 4});
            auto rot_mat = world_transform({0, 3}, {0, 3});
            copyTo(pos, tr_mat, {});
            Rodrigues(rot, rot_mat);
        }

        for (auto& opt : obj_pts) {
            auto pt = (Vec4f&)opt;
            pt[3] = 1.0f;

            pt = *(Vec4f*)Mat(inv_camera_transform * world_transform * pt).data;

            // 좌표계 변환
            pt[1] *= -1.0f;
            opt = (Vec3f&)pt;
        }

        // 오브젝트 포인트에 frustum culling 수행
        frustum_culling(90 * CV_PI / 180.0f, 60 * CV_PI / 180.0f, obj_pts);

        if (!obj_pts.empty()) {
            // 각 점을 매핑합니다.
            projectPoints(obj_pts, Vec3f(0, 0, 0), Vec3f(0, 0, 0), mat_cam, mat_disto, table_contours);

            // debug draw contours
            {
                vector<vector<Point>> contour_draw;
                auto& tbl = contour_draw.emplace_back();
                for (auto& pt : table_contours) { tbl.push_back({(int)pt[0], (int)pt[1]}); }
                drawContours(rgb, contour_draw, 0, {0, 0, 255}, 8);
            }
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
        xend = max(min(rgb.cols - 1, xend), xbeg);
        yend = max(min(rgb.rows - 1, yend), ybeg);
        ROI = Rect(xbeg, ybeg, xend - xbeg, yend - ybeg);
    }

    // ROI 존재 = 당구 테이블이 시야 내에 있음
    if (ROI.size().area() > 1000) {
        Mat4b roi_rgb = img.rgb(ROI);
        UMat roi_edge;
        mask(ROI).copyTo(roi_edge);
        Mat roi_mask(ROI.size(), roi_edge.type());
        roi_mask.setTo(0);

        // 현재 당구대 추정 위치 영역으로 마스크를 설정합니다.
        // 이미지 기반으로 하는 것보다 정확성은 다소 떨어지지만, 이미 당구대가 시야에서 벗어나 위치 추정이 어긋나기 시작하는 시점에서 정확성을 따질 겨를이 없습니다.
        {
            vector<vector<Point>> contours_;
            auto& contour = contours_.emplace_back();
            for (auto& pt : table_contours) { contour.emplace_back((int)pt[0] - ROI.x, (int)pt[1] - ROI.y); }

            // 컨투어 영역만큼의 마스크를 그립니다.
            drawContours(roi_mask, contours_, 0, {255}, FILLED);
        }

        // ROI 내에서, 당구대 영역을 재지정합니다.
        {
            UMat sub;

            // 당구대 영역으로 마스킹 수행
            roi_edge = roi_edge.mul(roi_mask);

            // 에지 검출 이전에, 팽창-침식 연산을 통해 에지를 단순화하고, 파편을 줄입니다.
            //auto iterations = m.ball.roi_smoothing_iteration_count;
            //dilate(roi_edge, roi_edge, {}, {-1, -1}, iterations);
            //erode(roi_edge, roi_edge, {}, {-1, -1}, iterations);
            GaussianBlur(roi_edge, roi_edge, {3, 3}, 15);
            threshold(roi_edge, roi_edge, 128, 255, THRESH_BINARY);
            show("roi_mask", roi_edge);

            // 내부에 닫힌 도형을 만들수 있게끔, 경계선을 깎아냅니다.
            roi_edge.row(0).setTo(0);
            roi_edge.row(roi_edge.rows - 1).setTo(0);
            roi_edge.col(0).setTo(0);
            roi_edge.col(roi_edge.cols - 1).setTo(0);

            erode(roi_edge, sub, {});
            bitwise_xor(roi_edge, sub, roi_edge);
            show("edge_new", roi_edge);
        }

        // 당구공 찾기 ... findContours 활용
        vector<vector<Point>> ball_contours;
        {
            vector<vector<Point>> candidates;
            vector<Vec4i> hierarchy;
            findContours(roi_edge, candidates, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            // 부모 있는 컨투어만 남깁니다.
            for (int idx = 0; idx < hierarchy.size(); ++idx) {
                auto [prev, next, child_first, parent] = hierarchy[idx].val;

                if (child_first == -1) {
                    ball_contours.emplace_back(move(candidates[idx]));
                }
            }

            // 디버그용 그리기 1
            // drawContours(roi_rgb, ball_contours, -1, {0, 255, 255}, 1);
            for (auto& ctr : ball_contours) {
                auto moment = moments(ctr);
                if (abs(moment.m00) < 1e-6) { continue; }

                auto cx = moment.m10 / moment.m00;
                auto cy = moment.m01 / moment.m00;
                auto dist_far = sqrt(moment.m00 / CV_PI);

                circle(roi_rgb, {(int)cx, (int)cy}, 3, {0, 0, 255}, -1);
                circle(roi_rgb, {(int)cx, (int)cy}, dist_far, {255, 255, 255}, 1);
            }
        }

        subtract(Scalar{255}, roi_mask, roi_mask);
        bitwise_xor(roi_rgb, roi_rgb, roi_rgb, roi_mask);
        show("roi", roi_rgb);
    }

    // 결과물 출력
    tick_tot.stop();
    float elapsed = tick_tot.getTimeMilli();
    putText(rgb, (stringstream() << "Elapsed: " << elapsed << " ms").str(), {0, rgb.rows - 5}, FONT_HERSHEY_PLAIN, 1.0, {255, 255, 255});
    show("source", rgb);
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

void recognizer_t::poll()
{
    auto& m = *impl_;
    decltype(m.img_show) shows;

    if (read_lock lock(m.img_show_mtx, try_to_lock); lock) {
        shows = move(m.img_show);
        m.img_show = {};
    }

    for (auto& pair : shows) {
        imshow(pair.first, pair.second);
    }
}
} // namespace billiards