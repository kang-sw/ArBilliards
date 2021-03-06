#pragma once
#include <opencv2/core/base.hpp>

#include "recognizer.hpp"

namespace billiards::pipes {
using namespace std::literals;

enum { N_MAX_LIGHT = 5 };

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
        PIPEPP_OPTION(render_3dresult_on_debug_mat, false);
        PIPEPP_OPTION(show_grid_representation, false,
                      u8"Matching 과정에서, 각 그리드의 대표 중심점에 해당하는 커널을"
                      " 디버그 화면에 렌더링합니다.");
        PIPEPP_OPTION(draw_final_ball_position, false,
                      u8"디버그 화면에 탐색된 공의 최종 위치를 렌더링합니다.");
        PIPEPP_OPTION(show_realtime_kernel, false);
        PIPEPP_OPTION(kernel_display_scale, 200u);
        PIPEPP_OPTION(suitability_map_scale, 1.0);
    };

    PIPEPP_CATEGORY(kernel, "Kernels")
    {
        PIPEPP_OPTION(ball_radius, 0.03029f, u8"공의 반지름 in metric",
                      pipepp::verify::minimum(1e-4f));
        PIPEPP_OPTION(n_dots, 1600u,
                      u8"커널에서 양으로 평가되는 점의 개수입니다."
                      " 음성 커널 또한 같은 개수로 생성됩니다.",
                      pipepp::verify::minimum(1u));
        PIPEPP_OPTION(random_seed, 42u);
        PIPEPP_OPTION(negative_weight_range, cv::Vec2f(1, 2),
                      u8"음의 가중치로 평가되는 커널 구간입니다. \n"
                      "조명이 계산되지 않는 별도의 커널이며, 전형적인 circleOp 연산을 통해 생성됩니다. \n"
                      "단, 거리 계산 공식은 조명이 계산된 양의 커널과 똑같이 적용합니다. \n"
                      "이 커널 구간에서 계산된 적합도는 음의 가중치로 적용되어, 전체 적합도를 감소시킵니다."
                      " 이를 통해 중심이 아닌 점에 대한 매칭 후보의 적합도를 떨어트려, 매칭을 중앙으로"
                      " 집중시킬 수 있게 됩니다.",
                      pipepp::verify::minimum_all<cv::Vec2f>(0.f)
                        | pipepp::verify::ascending<cv::Vec2f>());
        PIPEPP_OPTION(persp_linear_distance_amp, 1.f, u8"Perspective Projection 후 normalize된 커널의 증폭 계수");
    };

    PIPEPP_CATEGORY(colors, "Colors")
    {
        PIPEPP_OPTION(base_rgb, cv::Vec3f(1, 1, 1),
                      u8"공의 기본 RGB 색상입니다. 색상 계산 이후, 적합한 색공간으로 변환해 처리합니다.",
                      pipepp::verify::minimum_all<cv::Vec3f>(0.f));
        PIPEPP_OPTION(fresnel0, cv::Vec3f(0.05, 0.05, 0.05), "",
                      pipepp::verify::minimum_all<cv::Vec3f>(0.f));
        PIPEPP_OPTION(roughness, 0.05f, "", pipepp::verify::minimum(0.f));
        PIPEPP_OPTION(center_area_color_range_lo, cv::Vec3b(0, 0, 0),
                      u8"중심이 될 수 있는 색상 영역을 결정합니다. HSV 색공간 기준");
        PIPEPP_OPTION(center_area_color_range_hi, cv::Vec3b(0, 0, 0),
                      u8"중심이 될 수 있는 색상 영역을 결정합니다. HSV 색공간 기준");

        PIPEPP_CATEGORY(lights, "Lights")
        {
            PIPEPP_OPTION(n_lightings, 1u,
                          u8"조명의 개수입니다. 최대 5개 제한. \n"
                          "조명의 위치는 Unity 좌표계에서, 테이블에 대한 상대 위치입니다."
                          " 테이블의 상하 방향 인식은 항상 보장되지만, 전방과 후방 방향은 인식이 "
                          " 갱신될 때마다 새로 설정될 가능성이 존재하므로 가급적 조명은"
                          " Y축(종축)을 기준으로 대칭이 되게끔 배치해야 합니다.",
                          pipepp::verify::maximum(5u));
            PIPEPP_OPTION(ambient_rgb, cv::Vec3f(0.2, 0.2, 0.2),
                          u8"환경 조명의 밝기입니다.",
                          pipepp::verify::minimum_all<cv::Vec3f>(0.f));

            PIPEPP_CATEGORY(l0, "0")
            {
                PIPEPP_OPTION(pos, cv::Vec3f(0, 1, 0));
                PIPEPP_OPTION(rgb, cv::Vec3f(1, 1, 1), "", pipepp::verify::minimum_all<cv::Vec3f>(0.f));
            };
            PIPEPP_CATEGORY(l1, "1")
            {
                PIPEPP_OPTION(pos, cv::Vec3f(0, 1, 0));
                PIPEPP_OPTION(rgb, cv::Vec3f(1, 1, 1), "", pipepp::verify::minimum_all<cv::Vec3f>(0.f));
            };
            PIPEPP_CATEGORY(l2, "2")
            {
                PIPEPP_OPTION(pos, cv::Vec3f(0, 1, 0));
                PIPEPP_OPTION(rgb, cv::Vec3f(1, 1, 1), "", pipepp::verify::minimum_all<cv::Vec3f>(0.f));
            };
            PIPEPP_CATEGORY(l3, "3")
            {
                PIPEPP_OPTION(pos, cv::Vec3f(0, 1, 0));
                PIPEPP_OPTION(rgb, cv::Vec3f(1, 1, 1), "", pipepp::verify::minimum_all<cv::Vec3f>(0.f));
            };
            PIPEPP_CATEGORY(l4, "4")
            {
                PIPEPP_OPTION(pos, cv::Vec3f(0, 1, 0));
                PIPEPP_OPTION(rgb, cv::Vec3f(1, 1, 1), "", pipepp::verify::minimum_all<cv::Vec3f>(0.f));
            };
        };
    };

    PIPEPP_CATEGORY(match, "Matching")
    {
        PIPEPP_OPTION(color_space, "RGB"s,
                      u8"매칭이 수행되는 색공간입니다. 고정됨",
                      pipepp::verify::contains("RGB"s));
        PIPEPP_OPTION(error_base, 1.04f,
                      u8"에러 계산에 사용하는 지수의 밑입니다. 클수록 매칭이 엄격하게 평가됩니다.",
                      pipepp::verify::minimum(1.f));
        PIPEPP_OPTION(error_weight, cv::Vec3f(1, 1, 1),
                      u8"에러 계산 시, 색상의 각 채널에 적용할 에러 가중치입니다. 높을 수록 에러에 큰 영향을 줍니다.");
        PIPEPP_OPTION(negative_weight, 1.0f,
                      u8"Negative 커널의 픽셀에 얼마만큼의 음의 가중치를 부여할지 결정합니다.",
                      pipepp::verify::minimum(0.f));

        PIPEPP_CATEGORY(optimize, "Optimization")
        {
            PIPEPP_OPTION(grid_size, 64,
                          u8"이미지 정사각 그리드의 픽셀 크기입니다. "
                          " 각각의 그리드마다 새로 커널 조명을 갱신하므로, 그리드의 크기가 작을수록 계산량이 많아집니다.",
                          pipepp::verify::minimum(8));
            PIPEPP_OPTION(max_pixels_per_grid, 64u,
                          u8"하나의 그리드 내에 존재할 수 있는 중심 후보의 최대치입니다. 이를 넘어가는 픽셀은 임의로 삭제됩니다.",
                          pipepp::verify::minimum(1u));
            PIPEPP_OPTION(max_concurrent_kernels, 64u,
                          u8"동시에 실행되는 GPU 커널 루프의 개수입니다.",
                          pipepp::verify::minimum(1u));
            PIPEPP_OPTION(grid_area_threshold, 0.5f,
                          u8"그리드 영역 내에 존재하는, 유효 센터 픽셀의 비율이 어느 정도 이상일 때에만"
                          " 실제 커널 연산을 수행합니다. 이는 잡음이 섞인 그리드에서 매번 커널을 새로 계산하는"
                          " 낭비를 줄여줍니다.",
                          pipepp::verify::clamp(0.f, 1.f));
        };

        PIPEPP_OPTION(n_center_dilate, 6u,
                      u8"Center matrix의 설정 시 팽창 횟수입니다. 높일수록 더 많은 후보를 고려합니다."
                      " 팽창 연산이 먼저 적용되므로, 다수의 팽창 연산을 통해 조명 반사 등으로 인한 공의"
                      " 중심 후보 영역의 공백을 메움으로써 정밀도를 높일 수 있습니다.");
        PIPEPP_OPTION(n_center_erode, 6u,
                      u8"Center matrix의 설정 시 침식 횟수입니다. 높일수록 더 적은 후보를 고려합니다."
                      " 팽창 이후 적용되므로, 팽창으로 인해 고려하게 된 무의미한 픽셀들을 제외시킬 수 있습니다."
                      " 팽창 횟수보다 큰 값을 적용하면 성능이 향상되지만, valid한 중점이 지워질 수 있습니다.");
    };

    PIPEPP_CATEGORY(search, "Searching")
    {
        PIPEPP_OPTION(n_balls, 1, u8"찾아낼 공의 개수입니다.");
        PIPEPP_OPTION(next_ball_erase_amp, 1.0,
                      u8"다음 공을 찾기 위해 현재 공의 suitability를 지울 때,"
                      " 공 반경의 얼마만큼의 비율을 다음 공의 후보에서 제할지 결정합니다.");
        PIPEPP_OPTION(confidence_threshold, 0.25,
                      u8"유효한 공으로 인식되는 데 필요한 최소한의 컨피던스입니다.");
        PIPEPP_OPTION(conf_amp, 1.0f, u8"결과 컨피던스를 단순 배율로 조정합니다. ");
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
        imgproc::img_t const* p_imdesc;
        cv::Mat3b             debug_mat;
    };

    struct output_type {
        struct ball_position {
            // 찾아낸 공의 월드 좌표 및 확신도입니다.
            cv::Vec3f position;
            float     confidence;
        };

        // 다수의 출력을 포함할 수 있습니다.
        std::vector<ball_position> positions;
    };

    void        operator()(pipepp::execution_context& ec, input_type const& in, output_type& o);
    static void link(shared_data & sd, pipepp::execution_context & _exec_cont, input_type & i, pipepp::options & opt);

public:
    ball_finder_executor();
    ~ball_finder_executor();

private:
    void _update_kernel_by(pipepp::execution_context & ec, imgproc::img_t const& imdesc, cv::Vec3f loc_table_pos, cv::Vec3f loc_table_rot, cv::Vec3f loc_sample_pos);
    void _internal_loop(pipepp::execution_context & ec, input_type const& in, output_type& o);

private:
    struct impl;
    std::unique_ptr<impl> _m;
};

} // namespace billiards::pipes
