#include "recognition.hpp"
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <condition_variable>
#include <map>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/base.hpp>

using namespace std;

namespace billiards
{
using img_t = recognizer_t::parameter_type;
using img_cb_t = recognizer_t::process_finish_callback_type;
using opt_img_t = optional<img_t>;
using read_lock = shared_lock<shared_mutex>;
using write_lock = unique_lock<shared_mutex>;

/**
 * @note 클래스 내부에서 사용하는 월드 좌표계는 모두 Unity 좌표계로 통일합니다. 단, 카메라 좌표계는 opencv 좌표계입니다.
 */
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

    // cv::Vec3f table_points[4];
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

    static void cull_frustum(float hfov_rad, float vfov_rad, vector<cv::Vec3f>& obj_pts);
    void project_model(img_t const& img, vector<cv::Vec2f>&, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull = true) const;
    void transform_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, vector<cv::Vec3f>& model_vertexes) const;
    void recognizer_impl_t::get_world_transform_matx(cv::Vec3f pos, cv::Vec3f rot, cv::Mat& world_transform) const;
    static void get_camera_matx(img_t const& img, cv::Mat& mat_cam, cv::Mat& mat_disto);

    void get_table_model(std::vector<cv::Vec3f>& vertexes);
    static std::optional<cv::Mat> get_safe_ROI(cv::Mat const&, cv::Rect);

    /**
     * @param rvec oepncv 좌표계, 카메라 좌표계 기준 회전값입니다. 월드 회전으로 반환
     * @param tvec opencv 좌표계, 카메라 좌표계 기준 위치값입니다. 월드 이동으로 반환
     */
    void camera_to_world(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec) const;

    /**
     * 벡터를 로컬 좌표계에서 회전시킵니다.
     * @param rvec_target 회전시킬 회전 벡터
     * @param rvec_rotator 회전 정도
     */
    static cv::Vec3f rotate_local(cv::Vec3f rvec_target, cv::Vec3f rvec_rotator);

    void draw_axes(img_t const& img, cv::Mat& dest, cv::Vec3f rvec_world, cv::Vec3f tvec_world, float marker_length, int thickness = 2) const;
    void draw_circle(img_t const& img, cv::Mat& dest, float base_size, cv::Vec3f tvec_world, cv::Scalar color) const;
    // void proj_to_screen(img_t const& img, vector<cv::Vec3f> model_pt, )
};
} // namespace billiards
