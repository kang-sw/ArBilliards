#include "recognition.hpp"
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <condition_variable>
#include <iostream>
#include <map>
#include <set>
#include <boost/detail/container_fwd.hpp>
#include <boost/detail/container_fwd.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/affine.hpp>

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
    void find_table(img_t const& img, recognition_desc& desc, const cv::Mat& rgb, const cv::UMat& filtered, vector<cv::Vec2f>& contour_table);
};

void recognizer_impl_t::find_table(img_t const& img, recognition_desc& desc, const cv::Mat& rgb, const cv::UMat& filtered, vector<cv::Vec2f>& contour_table)
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
                    contour_table.push_back(Vec2f(pt.x, pt.y));
                }

                putText(rgb, (stringstream() << "[" << contour.size() << ", " << area_size << "]").str(), contour[0], FONT_HERSHEY_PLAIN, 1.0, {0, 255, 0});
            }
        }
    }

    if (contour_table.size() == 4) {
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
            for (auto& uv : contour_table) {
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
                rvec = tvec;
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
            assert(contour_table.size() == table_points_3d.size());

            // 오차를 감안해 공간에서 변의 길이가 table size의 mean보다 작은 값을 선정합니다.
            auto thres = sum(m.table.size)[0] * 0.5;
            for (int idx = 0; idx < contour_table.size() - 1; idx++) {
                auto& t = table_points_3d;
                auto& c = contour_table;
                auto len = norm(t[idx + 1] - t[idx]);

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
            solve_successful = solvePnP(obj_pts, contour_table, mat_cam, mat_disto, rvec, tvec, false, SOLVEPNP_IPPE);

            auto error_estimate = norm((tvec_estimate - tvec));

            if (error_estimate > 0.2) {
                tvec = tvec_estimate;
            } else {
                use_pnp_data = true;
            }
        }
        //*/

        if (solve_successful) {
            {
                cv::Vec2f sum = {};
                for (auto v : contour_table) { sum += v; }
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
            {
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

            // rvec의 validity를 테스트합니다.
            if (true) {
                auto& p = img.camera;
                double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

                double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
                auto mat_cam = cv::Mat(3, 3, CV_64FC1, M);
                auto mat_disto = cv::Mat(4, 1, CV_64FC1, disto);
                auto test_pts = obj_pts;
                Mat rot;
                Rodrigues(rvec, rot);
                for (auto& pt : test_pts) {
                    pt = *(Vec3d*)Mat(rot * (Vec3d)pt).data + *(Vec3d*)tvec.data;
                }
                vector<Vec2f> proj;
                projectPoints(test_pts, Vec3f::zeros(), Vec3f::zeros(), mat_cam, mat_disto, proj);

                vector<vector<Point>> contour_draw;
                auto& tbl = contour_draw.emplace_back();
                for (auto& pt : proj) { tbl.push_back({(int)pt[0], (int)pt[1]}); }
                drawContours(rgb, contour_draw, 0, {0, 255, 0}, 3);
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

            desc.table.confidence = 0.9f;
        }
    }
}

recognition_desc recognizer_impl_t::proc_img(img_t const& img)
{
    using namespace cv;

    recognition_desc desc = {};

    TickMeter tick_tot;
    tick_tot.start();
    auto& rgb = img.rgb;

    UMat uclor;
    img.rgb.copyTo(uclor);

    // 색공간 변환
    cvtColor(uclor, uclor, COLOR_RGBA2RGB);
    cvtColor(uclor, uclor, COLOR_RGB2YUV);
    show("yuv", uclor);

    // 색역 필터링 및 에지 검출
    UMat filtered;
    inRange(uclor, m.table.color_filter_min, m.table.color_filter_max, filtered);
    {
        UMat umat_temp;
        // dilate(filtered, umat_temp, {}, {-1, -1}, 4);
        // erode(umat_temp, filtered, {}, {-1, -1}, 4);
        erode(filtered, umat_temp, {});
        bitwise_xor(filtered, umat_temp, filtered);
    }
    show("filtered", filtered);

    // 테이블 위치 탐색
    vector<Vec2f> contour_table; // 2D 컨투어도 반환받습니다.
    find_table(img, desc, rgb, filtered, contour_table);
    auto mm = Mat(img.camera_transform);

    /* 당구공 위치 찾기 */
    // 1. 당구대 ROI를 추출합니다.
    // 2. 당구대 중점(desc.table... or filtered...) 및 노멀(항상 {0, 1, 0})을 찾습니다.
    // 3. 당구공의 uv 중점 좌표를 이용해 당구공과 카메라를 잇는 직선을 찾습니다.
    // 4. 위 직선을 투사하여, 당구대 평면과 충돌하는 지점을 찾습니다.
    // 5. PROFIT!

    // ROI 추출
    Rect rect_ROI;

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
            Vec3f rot(0, -table_yaw_flt, 0);
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

        // 각 점을 매핑합니다.
        projectPoints(obj_pts, Vec3f(0, 0, 0), Vec3f(0, 0, 0), mat_cam, mat_disto, contour_table);

        // debug draw contours
        {
            vector<vector<Point>> contour_draw;
            auto& tbl = contour_draw.emplace_back();
            for (auto& pt : contour_table) { tbl.push_back({(int)pt[0], (int)pt[1]}); }
            drawContours(rgb, contour_draw, 0, {0, 0, 255}, 3);
        }
    }

    {
        int xbeg, ybeg, xend, yend;
        xbeg = ybeg = numeric_limits<int>::max();
        xend = yend = -1;
    }

    // 결과물 출력
    tick_tot.stop();
    float elapsed = tick_tot.getTimeMilli();
    putText(rgb, (stringstream() << "Elpased: " << elapsed << " ms").str(), {0, rgb.rows - 5}, FONT_HERSHEY_PLAIN, 1.0, {255, 255, 255});
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