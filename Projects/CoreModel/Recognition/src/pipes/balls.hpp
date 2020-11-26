#pragma once
#include "recognizer.hpp"

namespace billiards::pipes {
using namespace std::literals;

/**
 *  공 탐색은 개괄적으로 다음과 같은 절차를 따릅니다.
 *
 *  0. 사전 설정 사항
 *      1. 커널 밀도 (circle 함수 반경, positive/negative 가중치 범위)
 *      2. 조명의 위치, 색상, 개수
 *
 *  (1. 옵션 변경 시)
 *      임의 개수의 3D 커널을 Z = 1m 거리에 공과 같은 반지름으로 생성합니다.
 *      커널 속성 = [[X, Y, Z], [R, G, B]]
 *
 *  1. 커널을 기본 색상으로 초기화
 *
 *  2. 조명 계산.
 *      공을 테이블 중점에 둔 것으로 가정하고, 조명에 의한 색상을 계산합니다.
 *      커널은 구형의 정규화된 정점 목록인데, 반대쪽 정점은 의미 없으므로 조명 계산 시 등지는
 *     정점은 모두 Z 값을 반전시켜줍니다.
 *      조명 계산이 끝난 커널은 Z값을 드랍하고(Orthogonal Projection), [X, Y], [R, G, B}만 남깁니다.
 *      이후 R, G, B 벡터 리스트는 원하는 색공간으로 변환
 *
 *  2. 컨볼루션
 *      투사점까지의 거리를 바탕으로 커널의 반경을 조정한 뒤, 후보 위치 개수 M by 커널 길이 N에 대해
 *     희소 커널 템플릿 매칭을 수행합니다. (색상 거리의 역수에 대한 지수함수)
 *
 *  3. 공 선택
 *      가장 높은 값을 반환하는 픽셀이 공의 중점이 되며, 여러 개의 공을 탐색하려는 경우 위의 가중치
 *     리스트에서 선택된 공의 반경 범위에 들어가는 모든 적합도 픽셀을 지운 뒤 다시 최대값을 찾습니다.
 *     (적합도 벡터를 iterate해 선택된 공의 반경 내에 들어가는 모든 픽셀을 지우면 될듯)
 *
 */
PIPEPP_EXECUTOR(ball_finder_executor)
{
    // 1. 공 개수
    // 2. 머터리얼 파라미터
    // 3. 반지름 등
    // 4. 기본 색상
    // 5. 커널 정점 개수
    PIPEPP_CATEGORY(debug, "Debug")
    {
        PIPEPP_OPTION(show_debug_mat, false);
        PIPEPP_OPTION(show_realtime_kernel, false);
    };

    PIPEPP_CATEGORY(kernel, "Kernels")
    {
        PIPEPP_OPTION(n_dots, 1600u,
                      u8"커널을 구성하는 점의 개수입니다. 많을수록 정밀하지만, 느려집니다.",
                      pipepp::verify::minimum(1u));
        PIPEPP_OPTION(positive_weight_range, cv::Vec2f(0, 1),
                      u8"양의 가중치로 평가되는 커널 구간입니다.\n"
                      "조명 계산 이후 커널 반경을 스케일하는 방식으로 동작하며, 따라서 "
                      " 공의 인식 반지름 자체를 넓히는 효과를 냅니다.",
                      pipepp::verify::minimum_all<cv::Vec2f>(0.f)
                        | pipepp::verify::ascending<cv::Vec2f>());
        PIPEPP_OPTION(negative_weight_range, cv::Vec2f(0, 1),
                      u8"음의 가중치로 평가되는 커널 구간입니다. \n"
                      "조명이 계산되지 않는 별도의 커널이며, 전형적인 circleOp 연산을 통해 생성됩니다. \n"
                      "단, 거리 계산 공식은 조명이 계산된 양의 커널과 똑같이 적용합니다. \n"
                      "이 커널 구간에서 계산된 적합도는 음의 가중치로 적용되어, 전체 적합도를 감소시킵니다."
                      " 이를 통해 중심이 아닌 점에 대한 매칭 후보의 적합도를 떨어트려, 매칭을 중앙으로"
                      " 집중시킬 수 있게 됩니다.",
                      pipepp::verify::minimum_all<cv::Vec2f>(0.f)
                        | pipepp::verify::ascending<cv::Vec2f>());
    };

    PIPEPP_CATEGORY(colors, "Colors")
    {
        PIPEPP_OPTION(base_rgb, cv::Vec3b(255, 255, 232),
                      u8"공의 기본 RGB 색상입니다. 색상 계산 이후, 적합한 색공간으로 변환해 처리합니다.");
    };

    PIPEPP_CATEGORY(match, "Matching")
    {
        PIPEPP_OPTION(color_space, "HSV"s,
                      u8"매칭이 수행되는 색공간입니다",
                      verify::color_space_string_verify);
        PIPEPP_OPTION(error_base, 1.04f,
                      u8"에러 계산에 사용하는 지수의 밑입니다. 클수록 매칭이 엄격하게 평가됩니다.",
                      pipepp::verify::minimum(1.f));
        PIPEPP_OPTION(error_weight, cv::Vec3f(1, 1, 1),
                      u8"에러 계산 시, 색상의 각 채널에 적용할 에러 가중치입니다. 높을 수록 에러에 큰 영향을 줍니다.");
        PIPEPP_OPTION(negative_weight, 1.0f,
                      u8"Negative 커널의 픽셀에 얼마만큼의 음의 가중치를 부여할지 결정합니다.",
                      pipepp::verify::minimum(0.f));
    };

    PIPEPP_CATEGORY(search, "Searching")
    {
        PIPEPP_OPTION(n_balls, 1,
                      u8"찾아낼 공의 개수입니다.");
    };

    struct input_type {
        // 중심이 될 수 있는 모든 점의 마스크입니다.
        // 전형적으로, 테이블 색상 필터의 반전을 테이블 전체 영역으로 마스킹합니다.
        cv::Mat1b center_area_mask;

        // 테이블 트랜스폼
        cv::Vec3f table_rot, table_pos;

        // 공의 색상을 계산할 도메인입니다.
        // 반드시 match::color_space에 지정된 색공간과 같은 값이어야 합니다.
        cv::Mat3f domain;

        // 기타 기본 파라미터
    };

    struct output_type
    {
        // 찾아낸 공의 월드 좌표 및 확신도입니다.
        cv::Vec3f position;
        float confidence;
    };    

    void operator()(pipepp::execution_context& ec, input_type& i, output_type& o);
};

} // namespace billiards::pipes
