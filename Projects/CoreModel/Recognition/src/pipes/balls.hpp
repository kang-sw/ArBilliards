#pragma once
#include "recognizer.hpp"

namespace billiards::pipes {
using namespace std::literals;

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
    };

    PIPEPP_CATEGORY(kernel, "Kernels")
    {
        PIPEPP_OPTION(n_dots, 1600u,
                      u8"Ŀ���� �����ϴ� ���� �����Դϴ�. �������� ����������, �������ϴ�.",
                      pipepp::verify::minimum(1u));
        PIPEPP_OPTION(positive_weight_range, cv::Vec2f(0, 1),
                      u8"���� ����ġ�� �򰡵Ǵ� Ŀ�� �����Դϴ�.\n"
                      "���� ��� ���� Ŀ�� �ݰ��� �������ϴ� ������� �����ϸ�, ���� "
                      " ���� �ν� ������ ��ü�� ������ ȿ���� ���ϴ�.",
                      pipepp::verify::minimum_all<cv::Vec2f>(0.f)
                        | pipepp::verify::ascending<cv::Vec2f>());
        PIPEPP_OPTION(negative_weight_range, cv::Vec2f(0, 1),
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
        PIPEPP_OPTION(base_rgb, cv::Vec3b(255, 255, 232),
                      u8"���� �⺻ RGB �����Դϴ�. ���� ��� ����, ������ ���������� ��ȯ�� ó���մϴ�.");
    };

    PIPEPP_CATEGORY(match, "Matching")
    {
        PIPEPP_OPTION(color_space, "HSV"s,
                      u8"��Ī�� ����Ǵ� �������Դϴ�",
                      verify::color_space_string_verify);
        PIPEPP_OPTION(error_base, 1.04f,
                      u8"���� ��꿡 ����ϴ� ������ ���Դϴ�. Ŭ���� ��Ī�� �����ϰ� �򰡵˴ϴ�.",
                      pipepp::verify::minimum(1.f));
        PIPEPP_OPTION(error_weight, cv::Vec3f(1, 1, 1),
                      u8"���� ��� ��, ������ �� ä�ο� ������ ���� ����ġ�Դϴ�. ���� ���� ������ ū ������ �ݴϴ�.");
        PIPEPP_OPTION(negative_weight, 1.0f,
                      u8"Negative Ŀ���� �ȼ��� �󸶸�ŭ�� ���� ����ġ�� �ο����� �����մϴ�.",
                      pipepp::verify::minimum(0.f));
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
    };

    struct output_type
    {
        // ã�Ƴ� ���� ���� ��ǥ �� Ȯ�ŵ��Դϴ�.
        cv::Vec3f position;
        float confidence;
    };    

    void operator()(pipepp::execution_context& ec, input_type& i, output_type& o);
};

} // namespace billiards::pipes
