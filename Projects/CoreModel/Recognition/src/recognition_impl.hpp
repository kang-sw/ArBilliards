#include "recognition.hpp"
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <condition_variable>
#include <map>
#include <unordered_map>
#include <vector>
#include <opencv2/calib3d.hpp>
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

    unordered_map<string, cv::Mat> img_show;
    unordered_map<string, cv::Mat> img_show_queue;
    shared_mutex img_show_mtx;

    recognition_desc prev_desc;
    cv::Vec3f table_pos_flt = {};
    cv::Vec3f table_rot_flt = {};

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
        img_show_queue[move(wnd_name)] = move(img);
    }

    /**
     * @addtogroup Recognition.Methods
     * 코드 정리를 위해 일회성으로 호출되는 메서드 목록
     *
     * @{
     */

    /**
     * 비동기 워커 스레드 콜백입니다.
     * 소멸자 호출 전까지 내부 루프를 반복합니다.
     */
    void async_worker_thread();

    /**
     * 주된 이미지 처리를 수행합니다.
     */
    recognition_desc proc_img(img_t const& img);

    /**
     * 테이블 위치를 찾습니다. 
     */
    void find_table(
      img_t const& img,
      recognition_desc& desc,
      const cv::Mat& rgb,
      const cv::UMat& filtered,
      vector<cv::Vec2f>& table_contours);

    /**
     * 마커로부터 테이블 위치를 보정합니다.
     */
    void correct_table_pos(
      img_t const& img,
      recognition_desc& desc,
      cv::Mat rgb,
      cv::Rect ROI,
      cv::Mat3b roi_rgb,
      vector<cv::Point> table_contour_partial);

    /**
     * 공의 위치를 찾습니다.
     */
    void find_ball_center(
      img_t const& img,
      vector<cv::Point> const& contours,
      struct ball_find_parameter_t const& params,
      struct ball_find_result_t& result);

    struct ball_find_result_t
    {
        cv::Point img_center;
        cv::Vec3f coord_center;
        float confidence;
    };

    struct ball_find_parameter_t
    {
        /** 카메라 원점의 좌표입니다. */
        cv::Vec3f camera_position;

        /** 광선을 투사할 테이블 평면입니다. */
        struct plane_t const& table_plane;
    };
    /** @} */

    /**
     * 내부적으로 필터링을 수행하는 테이블 위치 및 회전 설정자입니다.
     */
    cv::Vec3f set_filtered_table_pos(cv::Vec3f new_pos, float confidence = 1.0f);
    cv::Vec3f set_filtered_table_rot(cv::Vec3f new_rot, float confidence = 1.0f);

    /**
     * 각 정점에 대해, 시야 사각뿔에 대한 컬링을 수행합니다.
     * @param hfov_rad 라디안 단위의 수평 시야각
     * @param vfov_rad 라디안 단위의 수직 시야각
     * @param obj_pts 카메라 좌표계에 대한 다각형 포인트 목록입니다. 
     */
    static void cull_frustum(float hfov_rad, float vfov_rad, vector<cv::Vec3f>& obj_pts);

    /**
     * 대상 모델 버텍스 목록을 화면 상에 투영합니다.
     */
    static void project_model(img_t const& img, vector<cv::Vec2f>&, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull = true);

    /**
     * 대상 모델 버텍스 목록을 화면 상에 투영합니다. Point 벡터를 반환하는 편의 함수 버전입니다.
     */
    static void project_model(img_t const& img, vector<cv::Point>&, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull = true);

    /**
     * 대상 모델 버텍스 목록을 카메라 좌표계로 투영합니다.
     */
    static void transform_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, vector<cv::Vec3f>& model_vertexes);

    /**
     * 위치 벡터 및 로드리게스 회전 벡터로부터 월드 트랜스폼을 획득하는 헬퍼 함수입니다.
     */
    static void recognizer_impl_t::get_world_transform_matx(cv::Vec3f pos, cv::Vec3f rot, cv::Mat& world_transform);

    /**
     * 카메라 매트릭스를 획득합니다.
     */
    static void get_camera_matx(img_t const& img, cv::Mat& mat_cam, cv::Mat& mat_disto);

    /**
     * 테이블 크기로부터 모델 공간에서의 테이블에 대한 3D 원점을 획득합니다.
     * Y축이 0인 ZX 평면상에 위치합니다. 
     */
    static void get_table_model(std::vector<cv::Vec3f>& vertexes, cv::Vec2f model_size);

    /**
     * ROI를 검증하고, 주어진 mat 영역 내에 맞게 재계산합니다.
     */
    static bool get_safe_ROI_rect(cv::Mat const& mat, cv::Rect& roi);

    /**
     * 
     */
    static std::optional<cv::Mat> get_safe_ROI(cv::Mat const&, cv::Rect);

    /**
     * 화면 픽셀의 깊이로부터 3D 좌표를 계산합니다.
     */
    static void get_point_coord_3d(img_t const& img, float& io_x, float& io_y, float z);

    /**
     * Hue의 circular한 특성을 고려하여 HSV 필터링을 수행합니다.
     */
    static void filter_hsv(cv::InputArray input, cv::OutputArray output, cv::Vec3f, cv::Vec3f);

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

    /**
     * 3차원 축을 그립니다. 
     */
    void draw_axes(img_t const& img, cv::Mat& dest, cv::Vec3f rvec_world, cv::Vec3f tvec_world, float marker_length, int thickness = 2) const;

    /**
     * 3D 위치의 원을 화면에 투영합니다. 거리에 따라 화면 상에 나타나는 반경이 달라집니다.
     */
    void draw_circle(img_t const& img, cv::Mat& dest, float base_size, cv::Vec3f tvec_world, cv::Scalar color) const;
    // void proj_to_screen(img_t const& img, vector<cv::Vec3f> model_pt, )
};

// 평면을 나타내는 타입입니다.
struct plane_t
{
    cv::Vec3f N;
    float d;

    static plane_t from_NP(cv::Vec3f N, cv::Vec3f P)
    {
        N = cv::normalize(N);

        plane_t plane;
        plane.N = N;
        plane.d = 0.f;

        auto u = plane.calc_u(P, P + N).value();

        plane.d = -u;
        return plane;
    }

    float calc(cv::Vec3f const& pt) const
    {
        auto res = cv::sum(N.mul(pt))[0] + d;
        return abs(res) < 1e-6f ? 0 : res;
    }

    bool has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
    {
        return calc(P1) * calc(P2) < 0.f;
    }

    optional<float> calc_u(cv::Vec3f const& P1, cv::Vec3f const& P2) const
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

    optional<cv::Vec3f> find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
    {
        if (auto uo = calc_u(P1, P2)) {
            auto u = *uo;

            if (u <= 1.f && u >= 0.f) {
                return P1 + (P2 - P1) * u;
            }
        }
        return {};
    }
};

} // namespace billiards
