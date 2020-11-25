#pragma once
#include <opencv2/core/matx.hpp>
#pragma warning(disable : 4244 4305 4819)

#include "../image_processing.hpp"
#include "kangsw/spinlock.hxx"
#include "pipepp/pipepp.h"

namespace pipepp
{
namespace detail
{
class option_base;
}
} // namespace pipepp

namespace billiards::pipes
{
struct verify {
    inline static const auto color_space_string_verify = pipepp::verify::contains<std::string>("Lab", "YCrCb", "RGB", "YUV", "HLS", "HSV", "Luv");
};

auto build_pipe() -> std::shared_ptr<pipepp::pipeline<struct shared_data, struct input_resize>>;
} // namespace billiards::pipes

namespace billiards::pipes
{
struct ball_position_desc {
    using clock = std::chrono::system_clock;

    cv::Vec3f pos;
    cv::Vec3f vel;

    clock::time_point tp;
    double dt(clock::time_point now) const { return std::chrono::duration<double, clock::period>(now - tp).count(); }
    cv::Vec3f ps(clock::time_point now) const { return dt(now) * vel + pos; }
};

using ball_position_set = std::array<ball_position_desc, 4>;

struct shared_state {
    auto lock() const { return std::unique_lock{lock_}; }

    struct {
        cv::Vec3f pos = {}, rot = cv::Vec3f(1, 0, 0);
    } table;

    ball_position_set balls;

private:
    mutable kangsw::spinlock lock_;
};

struct shared_data : pipepp::base_shared_context {
    // options
    PIPEPP_DECLARE_OPTION_CLASS(shared_data);
    struct table {
        struct size {
            PIPEPP_DECLARE_OPTION_CATEGORY("Table.Size");
            PIPEPP_OPTION(outer, cv::Vec2d(1.8, 0.98));
            PIPEPP_OPTION(inner, cv::Vec2d(1.653, 0.823));
            PIPEPP_OPTION(fit, cv::Vec2d(1.735, 0.915));
        };
        struct filter {
            PIPEPP_DECLARE_OPTION_CATEGORY("Table.Filter");
            PIPEPP_OPTION(alpha_pos, 0.3);
            PIPEPP_OPTION(alpha_rot, 0.3);
            PIPEPP_OPTION(jump_threshold_distance, 0.1);

            PIPEPP_OPTION(color_lo, cv::Vec3b(175, 150, 60));
            PIPEPP_OPTION(color_hi, cv::Vec3b(10, 255, 255));
        };
        struct marker {
            PIPEPP_DECLARE_OPTION_CATEGORY("Table.Marker");

            PIPEPP_OPTION(count_x, 9);
            PIPEPP_OPTION(count_y, 5);
            PIPEPP_OPTION(felt_width, 1.735f);
            PIPEPP_OPTION(felt_height, 0.915f);
            PIPEPP_OPTION(dist_from_felt_long, 0.012f);
            PIPEPP_OPTION(dist_from_felt_short, 0.012f);
            PIPEPP_OPTION(step, 0.206f);
            PIPEPP_OPTION(width_shift_a, 0.0f);
            PIPEPP_OPTION(width_shift_b, 0.0f);
            PIPEPP_OPTION(height_shift_a, 0.0f);
            PIPEPP_OPTION(height_shift_b, 0.01f);
        };
    };

    PIPEPP_OPTION_AUTO(camera_FOV, cv::Vec2d(84.855, 53.27), "Common");

    struct ball {
        PIPEPP_OPTION_AUTO(radius, 0.030239439175, "Ball");
    };

    // data
public:
    std::shared_ptr<shared_state> state;

    recognizer_t::frame_desc imdesc_bkup;
    recognizer_t::process_finish_callback_type callback;

    cv::Mat debug_mat;

    cv::Mat cluster_color_mat;
    cv::Mat rgb, hsv;
    cv::UMat u_rgb, u_hsv;

    cv::Mat1b table_hsv_filtered;
    cv::Mat1b table_filtered_edge;


    struct cluster_type {
        cv::Mat1i label_2d_spxl;
        cv::Mat1i label_cluster_1darray; // super pixel의 대응되는 array 집합
    } cluster;

    struct {
        std::vector<cv::Vec2f> contour;

        cv::Vec3f pos, rot;
        float confidence;
    } table;

public:
    void get_marker_points_model(std::vector<cv::Vec3f>& model) const;
    

private:
    std::map<std::string, cv::Mat> converted_resources_;
};

struct input_resize {
    PIPEPP_DECLARE_OPTION_CLASS(input_resize);
    PIPEPP_OPTION_AUTO(desired_image_width, 1280);

    PIPEPP_OPTION_AUTO(show_sources, false, "Debug");
    PIPEPP_OPTION_AUTO(test_color_spaces, false, "Debug");

    using input_type = recognizer_t::frame_desc;
    struct output_type {
        cv::Size img_size;
        cv::Mat rgb;
        cv::Mat hsv;
        cv::UMat u_rgb;
        cv::UMat u_hsv;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void output_handler(pipepp::pipe_error, shared_data& sd, pipepp::execution_context& ec, output_type const& o);
};

struct contour_candidate_search {
    PIPEPP_DECLARE_OPTION_CLASS(contour_candidate_search);
    PIPEPP_OPTION_AUTO(table_color_filter_0_lo, cv::Vec3b(175, 150, 60), "Filter");
    PIPEPP_OPTION_AUTO(table_color_filter_1_hi, cv::Vec3b(10, 255, 255), "Filter");

    PIPEPP_OPTION_AUTO(show_debug_mat, false, "Debug");

    PIPEPP_OPTION_AUTO(area_threshold_ratio, 0.1, "Contour", "Minimul pixel size of table candidate area");
    PIPEPP_OPTION_AUTO(approx_epsilon_preprocess, 5.0, "Contour", "Epsilon value used for approximate table contours");
    PIPEPP_OPTION_AUTO(approx_epsilon_convex_hull, 1.0, "Contour", "Epsilon value used for approximate table contours after convex hull operation.");

    struct preprocess {
        PIPEPP_OPTION_AUTO(num_erode_prev, 2, "Preprocess", "Number of erode operation before apply dilate. Dilate count determined automatically. ");
        PIPEPP_OPTION_AUTO(num_erode_post, 6, "Preprocess", "Number of dilate operation after apply dilate. Dilate count is determined automatically.");
    };

    struct input_type {
        cv::Mat debug_display;
        cv::UMat u_hsv;
    };
    struct output_type {
        std::vector<cv::Vec2f> table_contour_candidate;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const&, input_resize::output_type const&, input_type&);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct output_pipe {
    PIPEPP_DECLARE_OPTION_CLASS(output_pipe);

    struct debug {
        PIPEPP_DECLARE_OPTION_CATEGORY("Debug");
        PIPEPP_OPTION(show_debug_mat, false);
    };

    using input_type = shared_data*;
    using output_type = nullptr_t;

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data& sd, input_type& i) { i = &sd; }
    static auto factory() { return pipepp::make_executor<output_pipe>(); }
};

} // namespace billiards::pipes