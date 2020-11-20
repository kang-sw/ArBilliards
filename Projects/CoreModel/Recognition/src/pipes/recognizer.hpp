#pragma once
#include <opencv2/core/matx.hpp>

#include "shared_data.hpp"
#include "../recognition.hpp"
#include "../image_processing.hpp"
#include "pipepp/pipeline.hpp"

namespace billiards::imgproc
{
using img_t = recognizer_t::parameter_type;

struct plane_t {
    cv::Vec3f N;
    float d;

    static plane_t from_NP(cv::Vec3f N, cv::Vec3f P);
    static plane_t from_rp(cv::Vec3f rvec, cv::Vec3f tvec, cv::Vec3f up);
    plane_t& transform(cv::Vec3f tvec, cv::Vec3f rvec);
    float calc(cv::Vec3f const& pt) const;
    bool has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const;
    std::optional<float> calc_u(cv::Vec3f const& P1, cv::Vec3f const& P2) const;
    std::optional<cv::Vec3f> find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const;
};

void filter_hsv(cv::InputArray input, cv::OutputArray output, cv::Vec3f min_hsv, cv::Vec3f max_hsv);
bool is_border_pixel(cv::Rect img_size, cv::Vec2i pixel, int margin = 3);
void get_table_model(std::vector<cv::Vec3f>& vertexes, cv::Vec2f model_size);
std::pair<cv::Matx33d, cv::Matx41d> get_camera_matx(billiards::recognizer_t::parameter_type const& img);
void cull_frustum_impl(std::vector<cv::Vec3f>& obj_pts, plane_t const* plane_ptr, size_t num_planes);
void cull_frustum(std::vector<cv::Vec3f>& obj_pts, std::vector<plane_t> const& planes);
void project_model_local(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes);
void project_points(std::vector<cv::Vec3f> const& points, cv::Matx33f const& camera, cv::Matx41f const& disto, std::vector<cv::Vec2f>& o_points);
cv::Matx44f get_world_transform_matx_fast(cv::Vec3f pos, cv::Vec3f rot);
void transform_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes);
void project_model_fast(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, cv::Vec3f obj_pos, cv::Vec3f obj_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes);
std::vector<plane_t> generate_frustum(float hfov_rad, float vfov_rad);
void project_model(img_t const& img, std::vector<cv::Vec2f>& mapped_contours, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h = 88, float FOV_v = 50);
void draw_axes(img_t const& img, cv::Mat const& dest, cv::Vec3f rvec, cv::Vec3f tvec, float marker_length, int thickness);
void camera_to_world(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec);
cv::Vec3f rotate_local(cv::Vec3f target, cv::Vec3f rvec);
cv::Vec3f set_filtered_table_rot(cv::Vec3f table_rot, cv::Vec3f new_rot, float alpha = 1.0f, float jump_threshold = FLT_MAX);
cv::Vec3f set_filtered_table_pos(cv::Vec3f table_pos, cv::Vec3f new_pos, float alpha = 1.0f, float jump_threshold = FLT_MAX);
void project_model(img_t const& img, std::vector<cv::Point>& mapped, cv::Vec3f obj_pos, cv::Vec3f obj_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h, float FOV_v);
void project_contours(img_t const& img, const cv::Mat& rgb, std::vector<cv::Vec3f> model, cv::Vec3f pos, cv::Vec3f rot, cv::Scalar color, int thickness, cv::Vec2f FOV_deg);
} // namespace billiards::imgproc

namespace billiards::pipes
{
auto build_pipe() -> std::shared_ptr<pipepp::pipeline<struct shared_data, struct input_resize>>;
}

namespace billiards::pipes
{
struct shared_state {
    struct {
        cv::Vec3f pos = {}, rot = cv::Vec3f(1, 0, 0);
    } table;
};

struct shared_data : pipepp::base_shared_context {
    // options
    PIPEPP_DEFINE_OPTION_CLASS(shared_data);
    PIPEPP_DEFINE_OPTION_2(table_size_outer, cv::Vec2d());
    PIPEPP_DEFINE_OPTION_2(table_size_inner, cv::Vec2d());
    PIPEPP_DEFINE_OPTION_2(table_size_fit, cv::Vec2d());

    // data
    std::shared_ptr<shared_state> state;

    recognizer_t::parameter_type param_bkup;
    recognizer_t::process_finish_callback_type callback;

    cv::Mat debug_mat;

    cv::Mat rgb, hsv;
    cv::UMat u_rgb, u_hsv;

    struct {
        std::vector<cv::Vec2f> contour;

        cv::Vec3f pos, rot;
        float confidence;
    } table;
};

struct input_resize {
    PIPEPP_DEFINE_OPTION_CLASS(input_resize);
    PIPEPP_DEFINE_OPTION_2(desired_image_width, 1280);

    PIPEPP_DEFINE_OPTION_2(debug_show_source, false);
    PIPEPP_DEFINE_OPTION_2(debug_show_hsv, false);

    using input_type = recognizer_t::parameter_type;
    struct output_type {
        cv::Size img_size;
        cv::Mat rgb;
        cv::Mat hsv;
        cv::UMat u_rgb;
        cv::UMat u_hsv;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void output_handler(pipepp::pipe_error, shared_data& sd, output_type const& o);
};

struct contour_candidate_search {
    PIPEPP_DEFINE_OPTION_CLASS(contour_candidate_search);
    PIPEPP_DEFINE_OPTION_2(table_color_filter_0_lo, cv::Vec3b(0, 0, 0));
    PIPEPP_DEFINE_OPTION_2(table_color_filter_1_hi, cv::Vec3b(180, 255, 255));

    PIPEPP_DEFINE_OPTION_2(debug_show_0_filtered, false, "debug");
    PIPEPP_DEFINE_OPTION_2(debug_show_1_edge, false, "debug");

    PIPEPP_DEFINE_OPTION_2(area_threshold_ratio, 0.2, "contour", "Minimul pixel size of table candidate area");
    PIPEPP_DEFINE_OPTION_2(approx_epsilon_preprocess, 5.0, "contour", "Epsilon value used for approximate table contours");
    PIPEPP_DEFINE_OPTION_2(approx_epsilon_convex_hull, 1.0, "contour", "Epsilon value used for approximate table contours after convex hull operation.");

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

struct table_edge_solver {
    PIPEPP_DEFINE_OPTION_CLASS(table_edge_solver);

    PIPEPP_DEFINE_OPTION_2(pnp_error_exp_fn_base,
                           1.06,
                           "PnP",
                           "Base of exponent function which is applied to calculate confidence of full-PNP solver result");
    PIPEPP_DEFINE_OPTION_2(pnp_conf_threshold,
                           1.06,
                           "PnP",
                           "Minimum required confidence value of full-PNP solver.");

    struct input_type {
        cv::Mat debug_mat;
        recognizer_t::parameter_type const* img_ptr;

        cv::Size img_size;

        std::vector<cv::Vec2f> const* table_contour;
        cv::Vec2f table_fit_size;

        cv::Vec3f table_pos_init;
        cv::Vec3f table_rot_init;
    };

    struct output_type {
        cv::Vec3f table_pos;
        cv::Vec3f table_rot;
        float confidence;
    };

    pipepp::pipe_error invoke(pipepp::execution_context& ec, input_type const& i, output_type& out);
    static void link_from_previous(shared_data const& sd, contour_candidate_search::output_type const& i, input_type& o);
};

} // namespace billiards::pipes