#pragma once
#include <opencv2/core/base.hpp>

#include "recognizer.hpp"

namespace billiards::pipes {
using namespace std::literals;

enum { N_MAX_LIGHT = 5 };

/**
 *  �� Ž���� ���������� ������ ���� ������ �����ϴ�.
 *
 *  0. ���� ���� ����
 *      1. Ŀ�� �е� (circle �Լ� �ݰ�, positive/negative ����ġ ����)
 *      2. ������ ��ġ, ����, ����
 *
 *  (1. �ɼ� ���� ��)
 *      ���� ������ 3D Ŀ���� Z = 1m �Ÿ��� ���� ���� ���������� �����մϴ�.
 *      Ŀ�� �Ӽ� = [[X, Y, Z], [R, G, B]]
 *
 *  1. Ŀ���� �⺻ �������� �ʱ�ȭ
 *
 *  2. ���� ���.
 *      ���� ���̺� ������ �� ������ �����ϰ�, ���� ���� ������ ����մϴ�.
 *      Ŀ���� ������ ����ȭ�� ���� ����ε�, �ݴ��� ������ �ǹ� �����Ƿ� ���� ��� �� ������
 *     ������ ��� Z ���� ���������ݴϴ�.
 *      ���� ����� ���� Ŀ���� Z���� ����ϰ�(Orthogonal Projection), [X, Y], [R, G, B}�� ����ϴ�.
 *      ���� R, G, B ���� ����Ʈ�� ���ϴ� ���������� ��ȯ
 *
 *  2. �������
 *      ������������ �Ÿ��� �������� Ŀ���� �ݰ��� ������ ��, �ĺ� ��ġ ���� M by Ŀ�� ���� N�� ����
 *     ��� Ŀ�� ���ø� ��Ī�� �����մϴ�. (���� �Ÿ��� ������ ���� �����Լ�)
 *
 *  3. �� ����
 *      ���� ���� ���� ��ȯ�ϴ� �ȼ��� ���� ������ �Ǹ�, ���� ���� ���� Ž���Ϸ��� ��� ���� ����ġ
 *     ����Ʈ���� ���õ� ���� �ݰ� ������ ���� ��� ���յ� �ȼ��� ���� �� �ٽ� �ִ밪�� ã���ϴ�.
 *     (���յ� ���͸� iterate�� ���õ� ���� �ݰ� ���� ���� ��� �ȼ��� ����� �ɵ�)
 *
 */
PIPEPP_EXECUTOR(ball_finder_executor)
{
    // 1. �� ����
    // 2. ���͸��� �Ķ����
    // 3. ������ ��
    // 4. �⺻ ����
    // 5. Ŀ�� ���� ����
    PIPEPP_CATEGORY(debug, "Debug")
    {
        PIPEPP_OPTION(show_debug_mat, false);
        PIPEPP_OPTION(show_realtime_kernel, false);
        PIPEPP_OPTION(kernel_display_scale, 200u);
    };

    PIPEPP_CATEGORY(kernel, "Kernels")
    {
        PIPEPP_OPTION(ball_radius, 0.03029f, u8"���� ������ in metric",
                      pipepp::verify::minimum(0.f));
        PIPEPP_OPTION(n_dots, 1600u,
                      u8"Ŀ�ο��� ������ �򰡵Ǵ� ���� �����Դϴ�."
                      " ���� Ŀ�� ���� ���� ������ �����˴ϴ�.",
                      pipepp::verify::minimum(1u));
        PIPEPP_OPTION(random_seed, 42u);
        PIPEPP_OPTION(positive_weight_range, cv::Vec2f(0, 1),
                      u8"���� ����ġ�� �򰡵Ǵ� Ŀ�� �����Դϴ�.\n"
                      "���� ��� ���� Ŀ�� �ݰ��� �������ϴ� ������� �����ϸ�, ���� "
                      " ���� �ν� ������ ��ü�� ������ ȿ���� ���ϴ�.\n"
                      "�ڵ����� 128�� ����� �����˴ϴ�.",
                      pipepp::verify::minimum_all<cv::Vec2f>(0.f)
                        | pipepp::verify::ascending<cv::Vec2f>());
        PIPEPP_OPTION(negative_weight_range, cv::Vec2f(1, 2),
                      u8"���� ����ġ�� �򰡵Ǵ� Ŀ�� �����Դϴ�. \n"
                      "������ ������ �ʴ� ������ Ŀ���̸�, �������� circleOp ������ ���� �����˴ϴ�. \n"
                      "��, �Ÿ� ��� ������ ������ ���� ���� Ŀ�ΰ� �Ȱ��� �����մϴ�. \n"
                      "�� Ŀ�� �������� ���� ���յ��� ���� ����ġ�� ����Ǿ�, ��ü ���յ��� ���ҽ�ŵ�ϴ�."
                      " �̸� ���� �߽��� �ƴ� ���� ���� ��Ī �ĺ��� ���յ��� ����Ʈ��, ��Ī�� �߾�����"
                      " ���߽�ų �� �ְ� �˴ϴ�.",
                      pipepp::verify::minimum_all<cv::Vec2f>(0.f)
                        | pipepp::verify::ascending<cv::Vec2f>());
    };

    PIPEPP_CATEGORY(colors, "Colors")
    {
        PIPEPP_OPTION(base_rgb, cv::Vec3f(1, 1, 1),
                      u8"���� �⺻ RGB �����Դϴ�. ���� ��� ����, ������ ���������� ��ȯ�� ó���մϴ�.",
                      pipepp::verify::minimum_all<cv::Vec3f>(0.f));
        PIPEPP_OPTION(fresnel0, cv::Vec3f(0.05, 0.05, 0.05), "",
                      pipepp::verify::minimum_all<cv::Vec3f>(0.f));
        PIPEPP_OPTION(roughness, 0.05f, "", pipepp::verify::minimum(0.f));
        PIPEPP_OPTION(center_area_color_range_lo, cv::Vec3b(0, 0, 0),
                      u8"�߽��� �� �� �ִ� ���� ������ �����մϴ�. HSV ������ ����");
        PIPEPP_OPTION(center_area_color_range_hi, cv::Vec3b(0, 0, 0),
                      u8"�߽��� �� �� �ִ� ���� ������ �����մϴ�. HSV ������ ����");

        PIPEPP_CATEGORY(lights, "Lights")
        {
            PIPEPP_OPTION(n_lightings, 1u,
                          u8"������ �����Դϴ�. �ִ� 5�� ����. \n"
                          "������ ��ġ�� Unity ��ǥ�迡��, ���̺� ���� ��� ��ġ�Դϴ�."
                          " ���̺��� ���� ���� �ν��� �׻� ���������, ����� �Ĺ� ������ �ν��� "
                          " ���ŵ� ������ ���� ������ ���ɼ��� �����ϹǷ� ������ ������"
                          " Y��(����)�� �������� ��Ī�� �ǰԲ� ��ġ�ؾ� �մϴ�.",
                          pipepp::verify::maximum(5u));
            PIPEPP_OPTION(ambient_rgb, cv::Vec3f(0.2, 0.2, 0.2),
                          u8"ȯ�� ������ ����Դϴ�.",
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
                      u8"��Ī�� ����Ǵ� �������Դϴ�. ������",
                      pipepp::verify::contains("RGB"s));
        PIPEPP_OPTION(error_base, 1.04f,
                      u8"���� ��꿡 ����ϴ� ������ ���Դϴ�. Ŭ���� ��Ī�� �����ϰ� �򰡵˴ϴ�.",
                      pipepp::verify::minimum(1.f));
        PIPEPP_OPTION(error_weight, cv::Vec3f(1, 1, 1),
                      u8"���� ��� ��, ������ �� ä�ο� ������ ���� ����ġ�Դϴ�. ���� ���� ������ ū ������ �ݴϴ�.");
        PIPEPP_OPTION(negative_weight, 1.0f,
                      u8"Negative Ŀ���� �ȼ��� �󸶸�ŭ�� ���� ����ġ�� �ο����� �����մϴ�.",
                      pipepp::verify::minimum(0.f));

        PIPEPP_CATEGORY(optimization, "Grid")
        {
            PIPEPP_OPTION(grid_size, 64, u8"�̹��� ���簢 �׸����� �ȼ� ũ���Դϴ�. ");
        };
    };

    PIPEPP_CATEGORY(search, "Searching")
    {
        PIPEPP_OPTION(n_balls, 1,
                      u8"ã�Ƴ� ���� �����Դϴ�.");
    };

    struct input_type {
        // �߽��� �� �� �ִ� ��� ���� ����ũ�Դϴ�.
        // ����������, ���̺� ���� ������ ������ ���̺� ��ü �������� ����ŷ�մϴ�.
        cv::Mat1b center_area_mask;

        // ���̺� Ʈ������
        cv::Vec3f table_rot, table_pos;

        // ���� ������ ����� �������Դϴ�.
        // �ݵ�� match::color_space�� ������ �������� ���� ���̾�� �մϴ�.
        cv::Mat3f domain;

        // ��Ÿ �⺻ �Ķ����
        imgproc::img_t const* p_imdesc;
    };

    struct output_type {
        struct ball_position {
            // ã�Ƴ� ���� ���� ��ǥ �� Ȯ�ŵ��Դϴ�.
            cv::Vec3f position;
            float confidence;
        };

        // �ټ��� ����� ������ �� �ֽ��ϴ�.
        std::vector<ball_position> positions;
    };

    void operator()(pipepp::execution_context& ec, input_type const& in, output_type& o);
    static void link(shared_data & sd, input_type & i, pipepp::options & opt);

public:
    ball_finder_executor();
    ~ball_finder_executor();

private:
    struct impl;
    std::unique_ptr<impl> _m;
};

} // namespace billiards::pipes
