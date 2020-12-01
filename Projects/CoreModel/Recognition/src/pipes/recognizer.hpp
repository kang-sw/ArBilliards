#pragma once
#include <opencv2/core/matx.hpp>
#pragma warning(disable : 4244 4305 4819)

#include "../image_processing.hpp"
#include "kangsw/spinlock.hxx"
#include "pipepp/pipepp.h"

namespace pipepp {
namespace detail {
class option_base;
}
} // namespace pipepp

namespace billiards::pipes {
struct verify {
    inline static const auto color_space_string_verify = pipepp::verify::contains<std::string>("Lab", "YCrCb", "RGB", "YUV", "HLS", "HSV", "Luv");
};

auto build_pipe() -> std::shared_ptr<pipepp::pipeline<struct shared_data, struct input_resize>>;
} // namespace billiards::pipes

namespace billiards::pipes {
struct ball_position_desc {
    using clock = std::chrono::system_clock;

    cv::Vec3f pos;
    cv::Vec3f vel;

    clock::time_point tp = clock::now();
    double            dt(clock::time_point now) const { return std::chrono::duration<double>(now - tp).count(); }
    cv::Vec3f         ps(clock::time_point now) const { return dt(now) * vel + pos; }
};

using ball_position_set = std::array<ball_position_desc, 4>;

struct shared_state {
    auto lock() const { return std::unique_lock{lock_}; }

    struct {
        cv::Vec3f pos = {}, rot = cv::Vec3f(1, 0, 0);
    } table;

    struct {
        cv::Vec3f pos = {}, rot = cv::Vec3f(1, 0, 0);
    } table_tr_context;

    ball_position_set balls;

private:
    mutable kangsw::spinlock lock_;
};

struct shared_data : pipepp::base_shared_context {
    // options
    PIPEPP_DECLARE_OPTION_CLASS(shared_data);

    PIPEPP_CATEGORY(debug, "Debug"){};

    PIPEPP_OPTION(camera_FOV, cv::Vec2d(84.855, 53.27));

    PIPEPP_CATEGORY(table, "Table")
    {
        PIPEPP_CATEGORY(size, "Size")
        {
            PIPEPP_OPTION(outer, cv::Vec2d(1.8, 0.98));
            PIPEPP_OPTION(inner, cv::Vec2d(1.653, 0.823));
            PIPEPP_OPTION(fit, cv::Vec2d(1.735, 0.915));
        };
        PIPEPP_CATEGORY(filter, "Filter")
        {
            PIPEPP_OPTION(alpha_pos, 0.3);
            PIPEPP_OPTION(alpha_rot, 0.3);
            PIPEPP_OPTION(jump_threshold_distance, 0.1);

            PIPEPP_OPTION(color_lo, cv::Vec3b(175, 150, 60));
            PIPEPP_OPTION(color_hi, cv::Vec3b(10, 255, 255));
        };
        PIPEPP_CATEGORY(marker, "Marker")
        {
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

    PIPEPP_CATEGORY(ball, "Ball")
    {
        PIPEPP_OPTION(radius, 0.030239439175);
        PIPEPP_OPTION(offset_from_table_plane, 0.002,
                      u8"테이블 평면 매칭은 당구대의 쿠션 상단을 기준으로 이루어지지만,"
                      " 실제 당구공은 테이블 평면보다 약간 아래쪽에 중심을 두고 있습니다."
                      " 이를 보정하기 위해 테이블을 Y축으로 평행이동하는 값입니다.");

        PIPEPP_CATEGORY(movement, "Movement")
        {
            PIPEPP_CATEGORY(correction, "Correction")
            {
                PIPEPP_OPTION(max_speed, 5.0f,
                              u8"미터 단위의 1초당 최대 이동 속도입니다. 속도가 이를 넘어서면,"
                              " 이전 페이즈의 이동을 다시 사용합니다. 순간적인 팝핑을 방지하기"
                              " 위한 프로퍼티");
                PIPEPP_OPTION(halted_tolerance, 0.01f,
                              u8"얼마 이상의 거리를 움직였을 때 이동으로 간주할지 결정하는 거리입니다."
                              " 해당 거리 안에서의 이동은 LPF에 의해 필터링됩니다.");
                PIPEPP_OPTION(halt_filter_alpha, 0.2f,
                              u8"공의 정지 상태 시 위치를 스무딩하는 LPF 알파 계수입니다.");
            };
        };
    };

    // data
public:
    recognizer_t::frame_desc                   imdesc_bkup;
    recognizer_t::process_finish_callback_type callback;

    cv::Mat debug_mat;

    cv::Mat  cluster_color_mat;
    cv::Mat  rgb, hsv;
    cv::UMat u_rgb, u_hsv;

    cv::Mat1b table_hsv_filtered;
    cv::Mat1b table_filtered_edge;

    struct cluster_type {
        cv::Mat1i label_2d_spxl;
        cv::Mat1i label_cluster_1darray; // super pixel의 대응되는 array 집합
    } cluster;

    struct {
        std::vector<cv::Vec2f> contour;
        cv::Vec3f              pos, rot;
        float                  confidence;
    } table;

public:
    void reload() override
    {
        converted_resources_.clear();
        table.pos = state_->table.pos;
        table.rot = state_->table.rot;

        for (auto& v : balls_) { v.second = 0.f; }
    }

    void    get_marker_points_model(std::vector<cv::Vec3f>& model) const;
    cv::Mat retrieve_image_in_colorspace(kangsw::hash_index hash);
    void    store_image_in_colorspace(kangsw::hash_index hash, cv::Mat v);
    void    update_ball_pos(size_t ball_idx, cv::Vec3f pos, float conf);
    auto    get_ball(size_t bidx) const -> ball_position_desc;
    auto    get_ball_raw(size_t bidx) const -> ball_position_desc;
    auto    get_ball_conf(size_t bidx) const { return balls_[bidx].second; }

    std::shared_ptr<shared_state> state_;

public:
    void _on_all_ball_gathered();

private:
    ball_position_set                    balls_prev_;
    std::pair<ball_position_desc, float> balls_[4];

    std::map<kangsw::hash_index, cv::Mat3b> converted_resources_;
    kangsw::spinlock                        lock_;
};

struct input_resize {
    PIPEPP_DECLARE_OPTION_CLASS(input_resize);
    PIPEPP_OPTION_AUTO(desired_image_width, 1280);

    PIPEPP_OPTION_AUTO(show_sources, false, "Debug");
    PIPEPP_OPTION_AUTO(test_color_spaces, false, "Debug");

    using input_type = recognizer_t::frame_desc;
    struct output_type {
        cv::Size img_size;
        cv::Mat  rgb;
        cv::Mat  hsv;
        cv::UMat u_rgb;
        cv::UMat u_hsv;
        float    resized_scale;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void        output_handler(pipepp::pipe_error, shared_data& sd, pipepp::execution_context& ec, output_type const& o);
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
        cv::Mat  debug_display;
        cv::UMat u_hsv;
    };
    struct output_type {
        std::vector<cv::Vec2f> table_contour_candidate;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void        link_from_previous(shared_data const&, input_resize::output_type const&, input_type&);
    static void        output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct output_pipe {
    PIPEPP_DECLARE_OPTION_CLASS(output_pipe);

    PIPEPP_CATEGORY(debug, "Debug")
    {
        PIPEPP_OPTION(show_debug_mat, false);
        PIPEPP_OPTION(render_debug_glyphs, true);
        PIPEPP_OPTION(table_top_view_scale, 640);
    };

    PIPEPP_CATEGORY(legacy, "Legacy")
    {
        PIPEPP_OPTION(setting_refresh_interval, 3.0f);

        PIPEPP_CATEGORY(unity, "Unity")
        {
            PIPEPP_OPTION(enable_table_depth_override, true);
            PIPEPP_OPTION(anchor_offset_vector, cv::Vec3f(0, 0, 0));

            PIPEPP_CATEGORY(phys, "Phys")
            {
                PIPEPP_OPTION(simulation_ball_radius, 0.0302);
                PIPEPP_OPTION(ball_restitution, 0.64);
                PIPEPP_OPTION(velocity_damping, 0.77);
                PIPEPP_OPTION(roll_coeff_on_contact, 0.77);
                PIPEPP_OPTION(roll_begin_time, 0.77);
                PIPEPP_OPTION(table_restitution, 0.77);
                PIPEPP_OPTION(table_roll_to_velocity_coeff, 0.77);
                PIPEPP_OPTION(table_velocity_to_roll_coeff, 0.77);
            };
        };
    };

    using input_type  = shared_data*;
    using output_type = nullptr_t;

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void        link_from_previous(shared_data& sd, input_type& i) { i = &sd; }
    static auto        factory() { return pipepp::make_executor<output_pipe>(); }

private:
    using clock = std::chrono::system_clock;
    clock::time_point latest_setting_refresh_;
};

} // namespace billiards::pipes