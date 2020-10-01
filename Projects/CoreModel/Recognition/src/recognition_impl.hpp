#include "recognition.hpp"
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <condition_variable>
#include <unordered_map>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <any>
#include <vector>

using namespace std;

namespace cv
{
template <int Size_, typename Ty_>
void to_json(nlohmann::json& j, const Vec<Ty_, Size_>& v)
{
    j = (std::array<Ty_, Size_>&)v;
}

template <int Size_, typename Ty_>
void from_json(const nlohmann::json& j, Vec<Ty_, Size_>& v)
{
    std::array<Ty_, Size_>& arr = j;
    v = (cv::Vec<Ty_, Size_>&)arr;
}

} // namespace cv

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

    vector<pair<string, double>> elapsed_seconds;

    recognition_desc prev_desc;
    cv::Vec3f table_pos_flt = {};
    cv::Vec3f table_rot_flt = {};

    cv::Vec3f ball_pos_prev[4] = {};

    // cv::Vec3f table_points[4];
    double table_yaw_flt = 0;

    unordered_map<string, any> vars, statics;

public:
    recognizer_impl_t(recognizer_t& owner)
        : m(owner)
    {
        worker_is_alive = true;
        worker = thread(&recognizer_impl_t::async_worker_thread, this);

        using nlohmann::json;
        using namespace cv;
        using namespace std;

        json params;
        params["fast-process-width"] = 540;

        {
            auto b = params["ball"];

            b["radius"] = 0.14 / CV_2PI;
            b["color"]["red"] = {Vec3f{115, 84, 0}, Vec3f{152, 255, 255}};
            b["color"]["orange"] = {Vec3f{75, 118, 0}, Vec3f{106, 255, 255}};
            b["color"]["white"] = {Vec3f{0, 0, 0}, Vec3f{81, 108, 255}};
        }

        {
            auto t = params["table"];

            t["size"]["fit"] = Vec2f{0.96f, 0.51f};
            t["size"]["outer"] = Vec2f(1.31f, 0.76f);
            t["size"]["inner"] = Vec2f(0.895f, 0.447f);

            t["filter"] = {Vec3f{0, 150, 0}, Vec3f{20, 255, 255}};
            t["hsv-shift"] = 15;
            t["cushion_height"] = 0.025;

            t["error-base"] = 1.02;

            t["LPF"]["position"] = 0.1666;
            t["LPF"]["rotation"] = 0.33;
        }
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
        show(move(wnd_name), img.getMat(cv::ACCESS_FAST));
    }

    void show(string wnd_name, cv::Mat img)
    {
        img_show_queue[move(wnd_name)] = img.clone();
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
    recognition_desc proc_img2(img_t const& img);

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
     * 주로 테이블에 활용 ...
     * 화면 상에서 폐색된 컨투어 리스트와, 모델의 시작 위치로부터 최종 위치를 추론합니다.
     */

    struct transform_estimation_param_t {
        int num_iteration = 10;
        int num_candidates = 64;
        float rot_axis_variant = 0.05;
        float rot_variant = 0.2f;
        float pos_initial_distance = 0.5f;
        int border_margin = 3;
        cv::Size2f FOV = {90, 60};
        float confidence_calc_base = 1.02f; // 에러 계산에 사용
        float iterative_narrow_ratio = 0.87f;

        cv::Mat debug_render_mat;
        bool render_debug_glyphs = true;
    };

    struct transform_estimation_result_t {
        cv::Vec3f position;
        cv::Vec3f rotation;
        float confidence;
    };

    static optional<transform_estimation_result_t> estimate_matching_transform(img_t const& img, vector<cv::Vec2f> const& input, vector<cv::Vec3f> model, cv::Vec3f init_pos, cv::Vec3f init_rot, transform_estimation_param_t const& param);

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

    struct ball_find_result_t {
        cv::Point img_center;
        cv::Point3f ball_position;
        float geometric_weight;
        float color_weight;
        float pixel_radius;
    };

    struct ball_find_parameter_t {
        struct plane_t const* table_plane; // 광선을 투사할 테이블 평면입니다. 반드시 카메라 좌표계
        cv::Mat precomputed_color_weights; // 미리 계산된 컬러 가중치 매트릭스.
        cv::Mat blue_mask;                 // 파란색 테이블 마스크
        cv::Rect ROI = {};
        cv::Mat rgb_debug;
        cv::Vec3f hsv_avg_filter_value; // 커널 계산에 사용할 색상 필터의 중간값
        int memoization_steps;
    };

    /**
     * 공의 이미지 상 중심을 찾습니다.
     */
    void find_ball_center(
      img_t const& img,
      vector<cv::Point> const& contours_src,
      struct ball_find_parameter_t const& params,
      struct ball_find_result_t& result);
    /** @} */

    /**
     * 최외각 경계를 1 픽셀 깎아냅니다.
     * binary 이미지로부터 erode를 통해 경계선 검출 시 경계선에 접한 컨투어가 닫힌 도형이 되도록 합니다.
     */
    static void carve_outermost_pixels(cv::InputOutputArray io, cv::Scalar as);

    /**
     * 내부적으로 필터링을 수행하는 테이블 위치 및 회전 설정자입니다.
     */
    cv::Vec3f set_filtered_table_pos(cv::Vec3f new_pos, float confidence = 1.0f);
    cv::Vec3f set_filtered_table_rot(cv::Vec3f new_rot, float confidence = 1.0f);

    /**
     * from 기준의 인덱스를 to 기준의 인덱스로 매핑합니다.
     * 종횡비는 일치하지 않아도 됩니다.
     */
    cv::Point map_index(cv::InputArray from, cv::InputArray to, cv::Point index);

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
    static void project_model(img_t const& img, vector<cv::Vec2f>&, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull = true, float FOV_h = 90, float FOV_v = 60);

    /**
     * 대상 모델 버텍스 목록을 화면 상에 투영합니다. Point 벡터를 반환하는 편의 함수 버전입니다.
     */
    static void project_model(img_t const& img, vector<cv::Point>&, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull = true, float FOV_h = 90, float FOV_v = 60);

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
    static std::pair<cv::Matx33d, cv::Matx41d> get_camera_matx(img_t const& img);

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
     * 카메라 좌표계 기준 3d 좌표에서 u, v를 계산합니다.
     */
    static array<float, 2> get_uv_from_3d(img_t const& img, cv::Point3f const& coord_3d);

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

    /**
     * 테이블 평면을 카메라 좌표계로 변환합니다.
     */
    static void recognizer_impl_t::plane_to_camera(img_t const& img, plane_t const& table_plane, plane_t& table_plane_camera);

    /**
     * metric 길이에 대한 해당 거리에서의 픽셀 크기를 반환합니다.
     */
    static float get_pixel_length(img_t const& img, float len_metric, float Z_metric);
};

// 평면을 나타내는 타입입니다.
struct plane_t {
    cv::Vec3f N;
    float d;

    static plane_t from_NP(cv::Vec3f N, cv::Vec3f P);
    plane_t& transform(cv::Vec3f tvec, cv::Vec3f rvec);
    float calc(cv::Vec3f const& pt) const;
    bool has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const;
    optional<float> calc_u(cv::Vec3f const& P1, cv::Vec3f const& P2) const;
    optional<cv::Vec3f> find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const;
};

enum BALL_INDEX
{
    BALL_RED,
    BALL_ORANGE,
    BALL_WHITE
};
} // namespace billiards
