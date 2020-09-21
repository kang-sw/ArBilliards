#pragma once
#include <memory>
#include <functional>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>


namespace billiards
{
/**
 * 당구대, 당구공, 큐대 전반적인 인식을 담당하는 클래스입니다.
 * 모든 거리 단위는 meter, 각도 단위는 degree입니다.
 * 
 */
class recognizer_t
{
public:
    /**
     * 생성자.
     * 즉시 백그라운드 스레드를 돌립니다.
     */
    recognizer_t();
    ~recognizer_t();

public: /* exposed properties */
    nlohmann::json props;

    /* 주요 영상처리 프로퍼티 */
    cv::Size actual_process_size = {960, 540};

    /* 당구공 관련 프로퍼티 */
    // 크기 및 색상을 설정 가능합니다.
    // red, white, orange의 경우 표기명으로, 공의 실제 색상과는 상이할 수 있습니다.

    // 당구공 인식 관련 프로퍼티 목록
    struct ball_param_type
    {
        float radius = 0.14 / CV_2PI;

        // 공의 Contour 목록을 파악한 후, 각각의 contour 다각형 면적이 최소 얼마 이상이 되어야 공으로 인식되는지 결정하는 값입니다.
        // 컨투어의 area_size(moment.m00)를 distance의 제곱으로 나눈 값을 아래 값과 비교합니다.
        float pixel_count_per_meter_min = 500;
        float pixel_count_per_meter_max = 10000;

        double edge_canny_thresholds[2] = {100, 50};

        // 에지 필터링에 사용할 각 색상 값
        struct
        {
            cv::Vec3f red[2] = {{115, 84, 0}, {152, 255, 255}};
            cv::Vec3f orange[2] = {{75, 118, 0}, {106, 255, 255}};
            cv::Vec3f white[2] = {{0, 0, 0}, {81, 108, 255}};
        } color;

        struct
        {
            int iterations = 7;                     // 전체 프로세스 반복 회수
            int num_candidates = 16;                // 한 번의 이터레이션에서 선택할 중점 후보의 개수
            float candidate_radius_amp = 2.0f;      // 중점 후보 선택시 반지름 증폭율
            int num_max_contours = 96;              // 한 번에 비교할 최대 컨투어 개수
            float weight_function_base = 1.03f;     // 가중치를 계산하는데 사용하는 지수함수의 밑
            float color_kernel_weight_base = 1.03f; // 커널에서 가중치를 계산하는데 사용하는 지수함수의 밑
            bool render_debug = true;               // 디버그 라인 그리기
            float memoization_distance_rate = 0.1f; // 메모이제이션 최적화 거리. 반지름에 대한 비
            float color_weight = 0.33f;             // 위치 대 색상 중 색상의 가중치
            float color_dist_amplitude = 64.0f;     // 가중치 계산 시, 정규화 구간의 색상을 증폭(클수록 가중치 낮아짐)
            float confidence_pivot_weight = 350.f;  // 컨피던스 1.0 기준 웨이트 합
        } search;

    } ball;

    /* 테이블 관련 프로퍼티 */
    struct table_param_type
    {
        cv::Vec2f recognition_size = {0.96f, 0.51f};
        cv::Vec2f outer_masking_size = {1.31f, 0.76f};
        cv::Vec2f inner_size = {0.895f, 0.447f};

        int color_cvt_type_rgb_to = cv::COLOR_RGB2HSV;
        cv::Scalar hsv_filter_min = {165, 150, 30};
        cv::Scalar hsv_filter_max = {5, 255, 255};
        float cushion_height = 0.025;

        // SolvePnP의 적용 이후, 추정된 회전과 위치값을 이용해 테이블 정점을 다시 화면에 투사하여, 검출된 정점과 택시캡 거리를 비교합니다. 이 때 오차가 가장 큰 점을 저장하여 아래 값과 비교합니다.
        int solvePnP_max_distance_error_threshold = 5;

        double polydp_approx_epsilon = 8;
        double min_pxl_area_threshold = 3e4; // 픽셀 넓이가 이보다 커야 당구대 영역으로 인식합니다.

        double LPF_alpha_pos = 0.1666;
        double LPF_alpha_rot = 0.33;

        // 이미지 내에서 '가깝다'고 판단하는 거리의 한계치입니다.
        // 1미터 거리당 픽셀 개수입니다.
        float pixel_distance_threshold_per_meter = 50;

        float aruco_detection_rect_radius_per_meter = 40;
        int aruco_dictionary = cv::aruco::DICT_4X4_50;
        cv::Point2f aruco_offset_from_corner = {0.035f, 0.035f};
        float aruco_marker_size = 0.0375f;
        int aruco_index_map[4] = {0, 1, 2, 3};
        std::array<cv::Vec3f, 4> aruco_horizontal_points{{
          {-(0.96f * 0.5f + 0.029f), 0, (0.51f * 0.5f + 0.029f)},
          {-(0.96f * 0.5f + 0.029f), 0, -(0.51f * 0.5f + 0.029f)},
          {(0.96f * 0.5f + 0.029f), 0, -(0.51f * 0.5f + 0.029f)},
          {(0.96f * 0.5f + 0.029f), 0, (0.51f * 0.5f + 0.029f)},
        }};
    } table;

    // 큐대 관련 프로퍼티
    // TODO: 효과적인 큐대 인식 방법 생각해보기

public:
    /* 카메라 파라미터 구조체 */
    struct camera_param_type
    {
        double fx, cx, fy, cy;
        double k1, k2, p1, p2;
    };

    /**
     * 공급 이미지 서술 구조체
     */
    struct parameter_type
    {
        cv::Vec3f camera_translation;
        cv::Vec4f camera_orientation; // In Euler angles ..
        cv::Matx<float, 4, 4> camera_transform;
        cv::Mat rgba;
        cv::Mat depth;

        camera_param_type camera;
    };

    using process_finish_callback_type = std::function<void(struct parameter_type const& image, struct recognition_desc const& result)>;

    /**
     * 인스턴스에 새 이미지를 공급합니다.
     * 백그라운드에서 실행 될 수 있으며, 실행 완료 후 재생할 콜백의 지정이 필요합니다.
     * 먼저 공급된 이미지의 처리가 시작되기 전에 새로운 이미지가 공급된 경우, 이전에 공급된 이미지는 버려집니다.
     */
    void refresh_image(parameter_type image, process_finish_callback_type&& callback = {});

    /**
     * 메인 스레드 루프입니다.
     * GUI를 띄우거나, 
     */
    void poll(std::unordered_map<std::string, cv::Mat>& shows);

    /**
     * 내부에 캐시된 이미지 인식 정보를 반환합니다.
     */
    struct recognition_desc const* get_recognition() const;

    /**
     * 내부에 캐시된 이미지를 반환합니다.
     */
    [[nodiscard]] parameter_type const* get_image() const;

private:
    std::unique_ptr<class recognizer_impl_t> impl_;
};

/**
 * 당구 게임의 인식 결과를 나타내는 구조체입니다.
 */
struct recognition_desc
{
    /**
     * 각 당구공의 인식 결과를, 확실한 정도와 함께 저장합니다.
     */
    struct ball_recognition_result
    {
        float position[3];
        float confidence;
    };

    union
    {
        ball_recognition_result balls[4];
        struct
        {
            ball_recognition_result red1;
            ball_recognition_result red2;
            ball_recognition_result orange;
            ball_recognition_result white;
        } ball;
    };

    /**
     * 테이블의 인식 결과입니다.
     */
    struct table_result
    {
        cv::Vec3f position;    // X Y Z in unity
        cv::Vec4f orientation; // Quaternion representation in unity coord
        float confidence;
    } table;

    /**
     * TODO: 큐대 인식 결과입니다.
     */
};

} // namespace billiards