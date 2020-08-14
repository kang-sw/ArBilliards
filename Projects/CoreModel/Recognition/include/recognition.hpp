#pragma once
#include <memory>
#include <functional>
#include <opencv2/core.hpp>
#include <optional>

namespace billiards
{
/**
 * 당구대, 당구공, 큐대 전반적인 인식을 담당하는 클래스입니다.
 * 모든 거리 단위는 meter, 각도 단위는 degree입니다.
 * 
 */
class recognizer
{
public: /* aliases */
    using process_finish_callback_type = std::function<void(struct image_feed const& image, struct recognition_desc const& result)>;

public: /* exposed properties */
    // 당구공 관련 프로퍼티
    // 크기 및 색상을 설정 가능합니다.
    // red, white, orange의 경우 표기명으로, 공의 실제 색상과는 상이할 수 있습니다.
    float ball_radious;        // 모든 당구공은 같은 크기를 갖는다 가정합니다.
    cv::Vec3f ball_red1_hsi;   // 붉은색 공의 색상 값입니다.
    cv::Vec3f ball_red2_hsi;   // 붉은색 공의 색상 값입니다.
    cv::Vec3f ball_white_hsi;  // 흰 공의     색상 값입니다.
    cv::Vec3f ball_orange_hsi; // 오렌지 공의 색상 값입니다.

    // 테이블 관련 프로퍼티
    cv::Vec2f table_size;
    cv::Vec3f table_hsi;

    // 큐대 관련 프로퍼티
    // TODO: 효과적인 큐대 인식 방법 생각해보기

public:
    /**
     * 공급 이미지 서술 구조체
     */
    struct parameter
    {
        cv::Vec3f camera_translation;
        cv::Vec3f camera_orientation; // In Euler angles ..
        cv::Mat rgb;
        cv::Mat depth;
    };

    /**
     * 인스턴스에 새 이미지를 공급합니다.
     * 백그라운드에서 실행 될 수 있으며, 실행 완료 후 재생할 콜백의 지정이 필요합니다.
     * 먼저 공급된 이미지의 처리가 시작되기 전에 새로운 이미지가 공급된 경우, 이전에 공급된 이미지는 버려집니다.
     */
    void refresh_image(parameter image, process_finish_callback_type callback = {});

    /**
     * 메인 스레드 루프입니다.
     * 주로 GUI를 띄우는 용도입니다.
     */
    void poll();

    /**
     * 내부에 캐시된 이미지 인식 정보를 반환합니다.
     */
    struct recognition_desc const* get_recognition() const;

    /**
     * 내부에 캐시된 이미지를 반환합니다.
     */
    parameter const* get_image() const;

private:
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
        cv::Vec3f position;
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
        } ball_name;
    };

    /**
     * 테이블의 인식 결과입니다.
     */
    struct table_result
    {
        cv::Vec3f position;
        cv::Vec3f normal;
        float confidence;
    } table;

    /**
     * TODO: 큐대 인식 결과입니다.
     */
};

} // namespace billiards