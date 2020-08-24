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
};

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

    // 컨투어 검출
    vector<Vec2f> contour_table;
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

            drawContours(rgb, candidates, idx, {0, 0, 255});

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

    // 검출된 컨투어에 대해 SolvePnp 적용
    if (contour_table.empty() == false) {
        thread_local static vector<Vec3f> obj_pts;
        thread_local static Mat tvec;
        thread_local static Mat rvec;
        auto& p = img.camera;
        double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

        double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
        auto mat_cam = cv::Mat(3, 3, CV_64FC1, M);
        auto mat_disto = cv::Mat(4, 1, CV_64FC1, disto);

        {
            float half_x = m.table.size.val[0] / 2;
            float half_z = m.table.size.val[1] / 2;

            // ref: OpenCV coordinate system
            // 참고: findContour로 찾아진 외각 컨투어 세트는 항상 반시계방향으로 정렬됩니다.
            obj_pts.clear();
            obj_pts.push_back({-half_x, 0, half_z});
            obj_pts.push_back({-half_x, 0, -half_z});
            obj_pts.push_back({half_x, 0, -half_z});
            obj_pts.push_back({half_x, 0, half_z});
        }

        // tvec의 초기값을 지정해주기 위해, 깊이 이미지를 이용하여 당구대 중점 위치를 추정합니다.
        bool estimation_valid = false;
        vector<Vec3d> table_points_3d = {};
        {
            // 카메라 파라미터는 컬러 이미지 기준이므로, 깊이 이미지 해상도를 일치시킵니다.
            Mat depth;
            resize(img.depth, depth, {img.rgb.cols, img.rgb.rows});

            auto& c = img.camera;

            // 당구대의 네 점의 좌표를 적절한 포인트 클라우드로 변환합니다.
            for (auto& uv : contour_table) {
                auto u = uv[0];
                auto v = uv[1];
                auto z_metric = depth.at<float>(v, u);

                auto& pt = table_points_3d.emplace_back();
                pt[2] = z_metric;
                pt[0] = z_metric * ((u - c.cx) / c.fx);
                pt[1] = z_metric * ((v - c.cy) / c.fy);
            }

            // 3d 포즈를추정해냅니다.
            Mat affine;
            Mat inliers;
            estimateAffine3D(obj_pts, table_points_3d, affine, inliers);

            // tvec은 평균치를 사용합니다.
            {
                Vec3d tvec_init = {};
                for (auto& pt : table_points_3d) {
                    tvec_init += pt;
                }
                tvec_init /= static_cast<float>(table_points_3d.size());
                tvec = Mat(tvec_init);
                rvec = tvec;
            }

            //tvec = affine.col(3);
            //auto obj_rotation = affine.colRange(0, 3);
            //Rodrigues(obj_rotation, rvec); // [0, 3)열을 rodrigues 표현식으로 전환

            //// 회전을 검증합니다.
            //{
            //    vector<Vec3d> table_point_set = table_points_3d;
            //    vector<double> dists;

            //    // 오브젝트의 모델 공간 포인트에 아핀 변환을 적용하고, 가장 오차가 적은 점을 찾습니다.
            //    for (auto const& obj_pt : obj_pts) {
            //        Vec4d tr = (Vec4f&)obj_pt;
            //        tr[3] = 1.0;

            //        tr = (Vec4d&)Mat(affine * tr).at<double>(0);

            //        double dist_min = numeric_limits<double>::max();
            //        decltype(table_point_set.begin()) min_it;
            //        for (auto it = table_point_set.begin();
            //             it != table_point_set.end();
            //             ++it) {
            //            auto dist_vec = (*it - (Vec3d&)tr);
            //            double dist = sqrt(dist_vec.dot(dist_vec));

            //            if (dist < dist_min) {
            //                dist_min = dist;
            //                min_it = it;
            //            }
            //        }

            //        dists.push_back(dist_min);
            //        table_point_set.erase(min_it);
            //    }

            //    double dist_avg = 0.0;
            //    for (auto& dist : dists) {
            //        dist_avg += dist;
            //    }

            //    dist_avg /= 4.0;
            //    cout << "info: average distances is " << dist_avg << "\n";
            //    estimation_valid = dist_avg < 0.05;
            //}
        }

        //*
        bool solve_successful = estimation_valid;
        /*/
        bool const solve_successful = solvePnP(obj_pts, contour_table, mat_cam, mat_disto, rvec, tvec, true, SOLVEPNP_IPPE);
        //*/

        if (solve_successful) {
            {
                Vec2f sum = {};
                for (auto v : contour_table) { sum += v; }
                Point draw_at = {int(sum.val[0] / 4), int(sum.val[1] / 4)};

                auto translation = Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                double dist = sqrt(translation.dot(translation));
                int thickness = max<int>(1, 2.0 / dist);

                putText(rgb, (stringstream() << Vec3f(dist)).str(), draw_at, FONT_HERSHEY_PLAIN, 2.0 / dist, {0, 0, 255}, thickness);
            }

            // tvec은 카메라 기준의 상대 좌표를 담고 있습니다.
            // img 구조체 인스턴스에는 카메라의 월드 트랜스폼이 담겨 있습니다.
            // tvec을 카메라의 orientation만큼 회전시키고, 여기에 카메라의 translation을 적용하면 물체의 월드 좌표를 알 수 있습니다.
            // rvec에는 카메라의 orientation을 곱해줍니다.
            {
                Vec4f pos = Vec4d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2), 0);
                pos[1] = -pos[1], pos[3] = 1.0;
                pos = img.camera_transform * pos;
                desc.table.position = (Vec3f&)pos;
            }

            // 테이블의 각 점을 월드 기준으로 바꿉니다.
            {
                
            }

            // 회전 또한 카메라 기준이므로, 트랜스폼을 적용합니다.
            if (false) {
                Mat cam_rotation = Mat(img.camera_transform)(Rect(0, 0, 3, 3));
                Mat obj_rotation;
                rvec.at<double>(1) *= -1; // 축 반전
                Rodrigues(rvec, obj_rotation);
                cam_rotation.convertTo(cam_rotation, CV_64FC1);

                // 두 회전을 순차 적용합니다.
                Mat rotation = cam_rotation * obj_rotation;
                Mat rotation_rodrigues;
                Rodrigues(rotation, rotation_rodrigues);

                desc.table.orientation = (Vec4d&)rotation_rodrigues.at<double>(0);
            }

            if (false) {
                Mat rod;
                Rodrigues(Mat(img.camera_transform)(Rect(0, 0, 3, 3)), rod);
                auto rod_vec = *(Vec4f*)rod.data;

                desc.table.orientation = rod_vec;
            }

            desc.table.confidence = 0.9f;
        }
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