#pragma once
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace pipepp {
namespace detail {
class pipeline_base;
}
} // namespace pipepp

namespace billiards {
/**
 * 당구대, 당구공, 큐대 전반적인 인식을 담당하는 클래스입니다.
 * 모든 거리 단위는 meter, 각도 단위는 degree입니다.
 * 
 */
class recognizer_t {
public:
    /**
     * 생성자.
     * 즉시 백그라운드 스레드를 돌립니다.
     */
    recognizer_t();
    ~recognizer_t();

    void initialize();
    void destroy();

public: /* exposed properties */
    nlohmann::json props;

    /* 주요 영상처리 프로퍼티 */
    cv::Size actual_process_size = {800, 480};
    cv::Size2f FOV = {84.9f, 52.9f};

public:
    /* 카메라 파라미터 구조체 */
    struct camera_param_type {
        double fx, cx, fy, cy;
        double k1, k2, p1, p2;
    };

    /**
     * 공급 이미지 서술 구조체
     */
    struct frame_desc {
        cv::Vec3f camera_translation;
        cv::Vec4f camera_orientation; // In Euler angles ..
        cv::Matx<float, 4, 4> camera_transform;
        cv::Mat rgba;
        cv::Mat depth;

        camera_param_type camera;
    };

    using process_finish_callback_type = std::function<void(struct frame_desc const& image, nlohmann::json const& result)>;

    /**
     * 인스턴스에 새 이미지를 공급합니다.
     * 백그라운드에서 실행 될 수 있으며, 실행 완료 후 재생할 콜백의 지정이 필요합니다.
     * 먼저 공급된 이미지의 처리가 시작되기 전에 새로운 이미지가 공급된 경우, 이전에 공급된 이미지는 버려집니다.
     */
    void refresh_image(frame_desc image, process_finish_callback_type&& callback = {});

    /**
     * 생성된 파이프라인 인스턴스를 반환합니다.
     * initialize() 호출 이후에만 valid합니다.
     */
    std::weak_ptr<pipepp::detail::pipeline_base> get_pipeline_instance() const;

    /**
     * 메인 스레드 루프입니다.
     */
    void poll(std::unordered_map<std::string, cv::Mat>& shows);

    /**
     * 내부에 캐시된 이미지 인식 정보를 반환합니다.
     */
    struct recognition_desc const* get_recognition() const;

    /**
     * 내부에 캐시된 이미지를 반환합니다.
     */
    [[nodiscard]] frame_desc get_image_snapshot() const;

    /**
     * 타이밍 리스트를 반환합니다.
     */
    std::vector<std::pair<std::string, std::chrono::microseconds>> get_latest_timings() const;

private:
    class implementation;
    std::unique_ptr<implementation> impl_;
};

} // namespace billiards

namespace std {
ostream& operator<<(ostream& strm, billiards::recognizer_t::frame_desc const& desc);
istream& operator>>(istream& strm, billiards::recognizer_t::frame_desc& desc);
} // namespace std