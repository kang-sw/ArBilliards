#pragma warning(disable : 4819)

#pragma once
#include "recognizer.hpp"

namespace billiards::pipes
{
class superpixel_executor
{
public:
    PIPEPP_DECLARE_OPTION_CLASS(superpixel_executor);
    PIPEPP_OPTION_AUTO(target_image_width, 1280, "Common", "", pipepp::verify::clamp(300, 8096));
    PIPEPP_OPTION_AUTO(color_space,
                       std::string("Lab"),
                       "Common",
                       R"(Input must be one of "Lab", "YCrCb", "RGB", "YUV", "HLS", "HSV")",
                       pipepp::verify::contains<std::string>("Lab", "YCrCb", "RGB", "YUV", "HLS", "HSV"));

    PIPEPP_OPTION_AUTO(true_SLIC_false_SEEDS, true, "flags");

    struct SEEDS {
        PIPEPP_OPTION_AUTO(num_iter, 4, "SEEDS");
    };
    PIPEPP_OPTION_AUTO(num_segments, 1024, "SEEDS");
    PIPEPP_OPTION_AUTO(num_levels, 5, "SEEDS");

    PIPEPP_OPTION_AUTO(show_segmentation_result, true, "debug");
    PIPEPP_OPTION_AUTO(segmentation_devider_color, cv::Vec3b(255, 0, 255), "debug");

    struct SLIC {
        PIPEPP_OPTION_AUTO(num_iter, 4, "SLIC");
    };
    PIPEPP_OPTION_AUTO(algo_index_SLICO_MSLIC_SLIC, 0, "SLIC");
    PIPEPP_OPTION_AUTO(region_size, 20, "SLIC");
    PIPEPP_OPTION_AUTO(ruler, 20, "SLIC");

    struct input_type {
        cv::Mat rgb;
    };

    struct output_type {
        cv::Mat3b resized_cluster_color_mat;
        cv::Mat1i labels;
        int num_labels;

        int color_convert_to;
        int color_convert_from;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& o);

public:
    superpixel_executor();
    ~superpixel_executor();

private:
    struct implmentation;
    std::unique_ptr<implmentation> impl_;
    int convert_to = -1, convert_from = -1;

public:
    struct link {
        PIPEPP_DECLARE_OPTION_CLASS(shared_data);
        PIPEPP_CATEGORY(preprocess, "Processing.Clustering.Preprocess")
        {
            PIPEPP_OPTION(enable_hsv_adjust, true);

            PIPEPP_OPTION(v_mult, 1.33);
            PIPEPP_OPTION(v_add, 100);

            PIPEPP_OPTION(enable_hsv_filter, false);
        };
    };
    static void link_from_previous(shared_data&, pipepp::execution_context&, input_type&);
    static void output_handler(pipepp::pipe_error e, shared_data& sd, output_type const& o);
};

class kmeans_executor
{
public:
    PIPEPP_DECLARE_OPTION_CLASS(kmeans_executor);
    struct debug {
        PIPEPP_DECLARE_OPTION_CATEGORY("Debug");
        PIPEPP_OPTION(show_kmeans_result, false);
        PIPEPP_OPTION(show_superpixel_result, false);
    };

    PIPEPP_OPTION_AUTO(weights_L_A_B_XY, (cv::Vec<float, 4>(1.0, 1.0, 1.0, 1.0)));

    struct kmeans {
        PIPEPP_DECLARE_OPTION_CATEGORY("k-means");
        PIPEPP_OPTION(flag_centoring_true_RANDOM_false_PP, false);

        PIPEPP_OPTION(N_cluster, 52, "Number of Clusters");
        PIPEPP_OPTION(attempts, 10);

        PIPEPP_OPTION(enabled, false);

        struct criteria {
            PIPEPP_DECLARE_OPTION_CATEGORY("Criteria");
            PIPEPP_OPTION(true_EPSILON_false_ITER, false);
            PIPEPP_OPTION(epsilon, 1.0);
            PIPEPP_OPTION(N_iter, 3, "", pipepp::verify::minimum(1));
        };
    };

public:
    struct input_type {
        superpixel_executor::output_type clusters;
    };

    using output_type = shared_data::cluster_type;

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& in, output_type& out);
    static void link_from_previous(shared_data const& sd, superpixel_executor::output_type const& o, input_type& i);
    static void output_handler(shared_data& sd, output_type const& o);

private:
    std::vector<cv::Vec<int32_t, 6>> spxl_sum;
    std::vector<cv::Vec<float, 5>> spxl_kmeans_param;
};

struct label_edge_detector {
    PIPEPP_DECLARE_OPTION_CLASS(label_edge_detector);
    PIPEPP_CATEGORY(debug, "Debug")
    {
        PIPEPP_OPTION(show_raw_edges, false);
    };

    PIPEPP_CATEGORY(edge, "Edge")
    {
        PIPEPP_OPTION(pp_dilate_erode_count, 0u,
                      u8"클러스터로부터 계산된 에지의 노이즈를 감소시키기 위해 "
                      "경계선을 N 번 팽창시킨 뒤 다시 침식합니다.");
    };

    struct input_type {
        cv::Mat1i labels;
    };

    struct output_type {
        cv::Mat1b edges;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& in, output_type& out);
};

/**
 * TODO: 클러스터 라벨 배열의 경계선 이미지를 획득하고, 여기서 hough 변환을 통해 모든 직선 후보를 찾습니다. 만나는 직선들로부터 모든 선분을 찾아내고, 선분으로부터 구성될 수 있는 모든 도형을 찾아냅니다. 이후 각 도형에 대해, 각 클러스터의 중심점을 iterate해, 테이블 색상과 가까운 클러스터가 가장 많이 포함되면서, 테이블 색상이 아닌 클러스터를 전혀 포함하지 않는 가장 큰 도형을 찾아냅니다. 이것이 테이블의 후보 사각형이 됩니다.
 */
class hough_line_executor
{
    PIPEPP_DECLARE_OPTION_CLASS(hough_line_executor);
    PIPEPP_CATEGORY(debug, "Debug")
    {
        PIPEPP_OPTION(show_source_image, false);
        PIPEPP_OPTION(show_found_lines, false);
    };

    PIPEPP_CATEGORY(hough, "Hough Lines")
    {
        PIPEPP_OPTION(use_P_version, false, u8"활성화 시 HoughLinesP를 대신 사용합니다.");
        PIPEPP_OPTION(rho, 1.0, "", pipepp::verify::clamp(0.0, 1.0));
        PIPEPP_OPTION(theta, 180.0, "", pipepp::verify::clamp(1e-3, 180.0));
        PIPEPP_OPTION(threshold, 1, "", pipepp::verify::minimum(0));

        PIPEPP_CATEGORY(np, "Non-P Version")
        {
            PIPEPP_OPTION(srn, 0.1, "", pipepp::verify::minimum(0.0));
            PIPEPP_OPTION(stn, 0.1, "", pipepp::verify::minimum(0.0));
        };
        PIPEPP_CATEGORY(p, "P Version")
        {
            PIPEPP_OPTION(min_line_len, 0.0, "", pipepp::verify::minimum(0.0));
            PIPEPP_OPTION(max_line_gap, 0.0, "", pipepp::verify::minimum(0.0));
        };
    };

public:
    struct input_type {
        cv::Mat1b edges;
        cv::Mat const* dbg_mat = {};
    };

    struct output_type {
        std::vector<cv::Vec3f> contours;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& in, output_type& out);
};

/**
 * 간단한 컨투어 필터입니다.
 * 마스크 데이터에서 보수적인 방법으로 컨투어를 획득합니다.
 */

} // namespace billiards::pipes