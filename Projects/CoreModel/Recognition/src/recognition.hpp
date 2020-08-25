#pragma once
#include <memory>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
    /* 당구공 관련 프로퍼티 */
    // 크기 및 색상을 설정 가능합니다.
    // red, white, orange의 경우 표기명으로, 공의 실제 색상과는 상이할 수 있습니다.
    float ball_radious = 0.042f; // 모든 당구공은 같은 크기를 갖는다 가정합니다.
    cv::Vec3f ball_red1_rgb;     // 붉은색 공의 색상 값입니다.
    cv::Vec3f ball_red2_rgb;     // 붉은색 공의 색상 값입니다.
    cv::Vec3f ball_white_rgb;    // 흰 공의     색상 값입니다.
    cv::Vec3f ball_orange_rgb;   // 오렌지 공의 색상 값입니다.

    /* 테이블 관련 프로퍼티 */
    struct table_param_type
    {
        cv::Vec2f size = {0.96f, 0.51f};
        int color_cvt_type_rgb_to = cv::COLOR_RGB2HSV;
        cv::Scalar color_filter_min = {0, 90, 150};
        cv::Scalar color_filter_max = {130, 140, 255};

        double polydp_approx_epsilon = 10;
        double min_pxl_area_threshold = 2e4; // 픽셀 넓이가 이보다 커야 당구대 영역으로 인식합니다.

        double LPF_alpha_pos = 0.03;
        double LPF_alpha_rot = 0.03;
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