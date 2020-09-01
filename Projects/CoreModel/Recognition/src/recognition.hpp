#pragma once
#include <memory>
#include <functional>
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
    /* 주요 영상처리 프로퍼티 */
    cv::Size actual_process_size = {960, 540};

    /* 당구공 관련 프로퍼티 */
    // 크기 및 색상을 설정 가능합니다.
    // red, white, orange의 경우 표기명으로, 공의 실제 색상과는 상이할 수 있습니다.

    // 당구공 인식 관련 프로퍼티 목록
    struct ball_param_type
    {
        float radius = 0.014 / CV_2PI;
        cv::Vec3f red1_rgb;
        cv::Vec3f red2_rgb;
        cv::Vec3f white_rgb;
        cv::Vec3f orange_rgb;

        // 당구대 ROI의 마스크를 스무싱하는 데 사용하는 파라미터입니다.
        // 팽창-침식 연산을 통해 파편을 제거하며, 이 값은 이터레이션 횟수를 정의합니다.
        int roi_smoothing_iteration_count = 4;
    } ball;

    /* 테이블 관련 프로퍼티 */
    struct table_param_type
    {
        cv::Vec2f recognition_size = {0.96f, 0.51f};
        cv::Vec2f outer_masking_size = {1.31f, 0.76f};
        cv::Vec2f inner_size = {0.895f, 0.447f};

        int color_cvt_type_rgb_to = cv::COLOR_RGB2HSV;
        cv::Scalar sv_filter_min = {0, 150, 30};
        cv::Scalar sv_filter_max = {255, 255, 255};
        float cushion_height = 0.025;
        int hue_filter_min = 165;
        int hue_filter_max = 5;

        double polydp_approx_epsilon = 8;
        double min_pxl_area_threshold = 2e4; // 픽셀 넓이가 이보다 커야 당구대 영역으로 인식합니다.

        double LPF_alpha_pos = 0.66;
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
        cv::Mat rgb;
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
    void poll();

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
            ball_recognition_result white;
            ball_recognition_result orange;
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