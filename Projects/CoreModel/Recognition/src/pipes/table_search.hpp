#pragma once
#include "recognizer.hpp"

namespace billiards::pipes
{
class superpixel
{
public:
    PIPEPP_DECLARE_OPTION_CLASS(superpixel);
    PIPEPP_OPTION(target_image_width, 1280, "Common");
    PIPEPP_OPTION(color_space,
                  std::string("Lab"),
                  "Common",
                  R"(Input must be one of "Lab", "YCrCb", "RGB", "YUV", "HLS", "HSV")",
                  ([](std::string& str) {
                      static const std::set<std::string> valids = {"Lab", "YCrCb", "RGB", "YUV", "HLS", "HSV"};
                      if (valids.contains(str) == false) { return str = *valids.begin(), false; }
                      return true;
                  }));

    PIPEPP_OPTION(true_SLIC_false_SEEDS, true, "flags");

    struct SEEDS {
        PIPEPP_OPTION(num_iter, 4, "SEEDS");
    };
    PIPEPP_OPTION(num_segments, 1024, "SEEDS");
    PIPEPP_OPTION(num_levels, 5, "SEEDS");

    PIPEPP_OPTION(show_segmentation_result, true, "debug");
    PIPEPP_OPTION(segmentation_devider_color, cv::Vec3b(255, 0, 255), "debug");

    struct SLIC {
        PIPEPP_OPTION(num_iter, 4, "SLIC");
    };
    PIPEPP_OPTION(algo_index_SLICO_MSLIC_SLIC, 0, "SLIC");
    PIPEPP_OPTION(region_size, 20, "SLIC");
    PIPEPP_OPTION(ruler, 20, "SLIC");

    struct input_type {
        cv::Mat rgb;
    };

    struct output_type {
        cv::Mat3b resized_cluster_color_mat;
        cv::Mat1i labels;

        int color_convert_to;
        int color_convert_from;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& o);

public:
    superpixel();
    ~superpixel();

private:
    struct implmentation;
    std::unique_ptr<implmentation> impl_;
    int convert_to = -1, convert_from = -1;

public:
    struct link {
        PIPEPP_DECLARE_OPTION_CLASS(shared_data);
        struct preprocess {
            PIPEPP_DECLARE_OPTION_CATEGORY("Processing.Clustering.Preprocess");
            PIPEPP_CATEGORY_OPTION(enable_hsv_adjust, true);

            PIPEPP_CATEGORY_OPTION(v_mult, 1.33);
            PIPEPP_CATEGORY_OPTION(v_add, 100);
        };
    };
    static void link_from_previous(shared_data&, pipepp::execution_context&, input_type&);
    static void output_handler(pipepp::pipe_error e, shared_data& sd, output_type const& o);
};

class clustering
{
public:
    PIPEPP_DECLARE_OPTION_CLASS(clustering);
    struct debug {
        PIPEPP_DECLARE_OPTION_CATEGORY("Debug");
        PIPEPP_CATEGORY_OPTION(show_kmeans_result, false);
        PIPEPP_CATEGORY_OPTION(show_superpixel_result, false);
    };

    PIPEPP_OPTION(weights_L_A_B_XY, (cv::Vec<float, 4>(1.0, 1.0, 1.0, 1.0)));

    struct kmeans {
        PIPEPP_DECLARE_OPTION_CATEGORY("k-means");
        PIPEPP_CATEGORY_OPTION(flag_enable_cluster_count_estimation, false, "Automatically calculates number of clusters.");
        PIPEPP_CATEGORY_OPTION(flag_centoring_true_RANDOM_false_PP, false);

        PIPEPP_CATEGORY_OPTION(N_cluster, 52, "Number of Clusters");
        PIPEPP_CATEGORY_OPTION(attempts, 10);

        PIPEPP_CATEGORY_OPTION(enabled, false);

        struct criteria {
            PIPEPP_DECLARE_OPTION_CATEGORY("Criteria");
            PIPEPP_CATEGORY_OPTION(true_EPSILON_false_ITER, false);
            PIPEPP_CATEGORY_OPTION(epsilon, 1.0);
            PIPEPP_CATEGORY_OPTION(N_iter, 3);
        };
    };

public:
    struct input_type {
        superpixel::output_type clusters;
    };

    using output_type = shared_data::cluster_type;

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& in, output_type& out);
    static void link_from_previous(shared_data const& sd, superpixel::output_type const& o, input_type& i);

private:
    std::vector<cv::Vec<int32_t, 6>> spxl_sum;
    std::vector<cv::Vec<float, 5>> spxl_kmeans_param;
};

/**
 * TODO: 클러스터 라벨 배열의 경계선 이미지를 획득하고, 여기서 hough 변환을 통해 모든 직선 후보를 찾습니다. 만나는 직선들로부터 모든 선분을 찾아내고, 선분으로부터 구성될 수 있는 모든 도형을 찾아냅니다. 이후 각 도형에 대해, 각 클러스터의 중심점을 iterate해, 테이블 색상과 가까운 클러스터가 가장 많이 포함되면서, 테이블 색상이 아닌 클러스터를 전혀 포함하지 않는 가장 큰 도형을 찾아냅니다. 이것이 테이블의 후보 사각형이 됩니다.
 */
class table_line_finding
{
};

} // namespace billiards::pipes