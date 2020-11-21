#pragma once
#include <optional>
#include <opencv2/opencv.hpp>
#include <random>
#include "recognition.hpp"

namespace billiards::imgproc
{
using img_t = recognizer_t::frame_desc;

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
auto get_camera_matx(billiards::recognizer_t::frame_desc const& img) -> std::pair<cv::Matx33d, cv::Matx41d>;
void cull_frustum_impl(std::vector<cv::Vec3f>& obj_pts, plane_t const* plane_ptr, size_t num_planes);
void cull_frustum(std::vector<cv::Vec3f>& obj_pts, std::vector<plane_t> const& planes);
void project_model_local(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes);
void project_points(std::vector<cv::Vec3f> const& points, cv::Matx33f const& camera, cv::Matx41f const& disto, std::vector<cv::Vec2f>& o_points);
auto get_world_transform_matx_fast(cv::Vec3f pos, cv::Vec3f rot) -> cv::Matx44f;
void transform_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes);
void project_model_fast(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, cv::Vec3f obj_pos, cv::Vec3f obj_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes);
auto generate_frustum(float hfov_rad, float vfov_rad) -> std::vector<plane_t>;
void project_model(img_t const& img, std::vector<cv::Vec2f>& mapped_contours, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h = 88, float FOV_v = 50);
void draw_axes(img_t const& img, cv::Mat const& dest, cv::Vec3f rvec, cv::Vec3f tvec, float marker_length, int thickness);
void camera_to_world(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec);
auto rotate_local(cv::Vec3f target, cv::Vec3f rvec) -> cv::Vec3f;
auto set_filtered_table_rot(cv::Vec3f table_rot, cv::Vec3f new_rot, float alpha = 1.0f, float jump_threshold = FLT_MAX) -> cv::Vec3f;
auto set_filtered_table_pos(cv::Vec3f table_pos, cv::Vec3f new_pos, float alpha = 1.0f, float jump_threshold = FLT_MAX) -> cv::Vec3f;
void project_model(img_t const& img, std::vector<cv::Point>& mapped, cv::Vec3f obj_pos, cv::Vec3f obj_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h, float FOV_v);
void project_contours(img_t const& img, const cv::Mat& rgb, std::vector<cv::Vec3f> model, cv::Vec3f pos, cv::Vec3f rot, cv::Scalar color, int thickness, cv::Vec2f FOV_deg);
bool get_safe_ROI_rect(cv::Mat const& mat, cv::Rect& roi);
float contour_distance(std::vector<cv::Vec2f> const& ct_a, std::vector<cv::Vec2f>& ct_b);
void plane_to_camera(img_t const& img, plane_t const& table_plane, plane_t& table_plane_camera);
void get_point_coord_3d(img_t const& img, float& io_x, float& io_y, float z_metric);
auto get_uv_from_3d(img_t const& img, cv::Point3f const& coord_3d) -> std::array<float, 2>;
float get_pixel_length(img_t const& img, float len_metric, float Z_metric);
int get_pixel_length_on_contact(img_t const& imdesc, plane_t plane, cv::Point pt, float length);

struct transform_estimation_param_t {
    int num_iteration = 10;
    int num_candidates = 64;
    float rot_axis_variant = 0.05;
    float rot_variant = 0.2f;
    float pos_initial_distance = 0.5f;
    int border_margin = 3;
    cv::Size2f FOV = {90, 60};
    float confidence_calc_base = 1.02f; // 에러 계산에 사용
    float iterative_narrow_ratio = 0.6f;

    cv::Mat debug_render_mat;
    bool render_debug_glyphs = true;
    bool do_parallel = true;

    cv::Rect contour_cull_rect;
};

struct transform_estimation_result_t {
    cv::Vec3f position;
    cv::Vec3f rotation;
    float confidence;
};

std::optional<transform_estimation_result_t> estimate_matching_transform(img_t const& img, std::vector<cv::Vec2f> const& input_param, std::vector<cv::Vec3f> model, cv::Vec3f init_pos, cv::Vec3f init_rot, transform_estimation_param_t const& p);

template <typename Vec_>
void drag_contour(std::vector<Vec_>& points, Vec_ center, double direction)
{
    // 거리의 최장점을 계산
    auto max_it = std::max_element(
      points.begin(),
      points.end(),
      [&center](Vec_ const& a, Vec_ const& b) { return cv::norm(a - center, cv::NORM_L2SQR) < cv::norm(b - center, cv::NORM_L2SQR); });

    if (max_it == points.end())
        return;

    double max_dist = cv::norm(*max_it - center);
    if (max_dist != 0) {
        direction /= max_dist;

        for (auto& pt : points) {
            auto delta_vect = pt - center;
            pt += delta_vect * direction;
        }
    }
}

template <typename Ty_>
cv::Matx<Ty_, 3, 3> rodrigues(cv::Vec<Ty_, 3> v)
{
    // cv::Matx<Ty_, 3, 3> retmat;
    // cv::Rodrigues(v, retmat);
    // return retmat;

    using mat_t = cv::Matx<Ty_, 3, 3>;

    auto O = cv::norm(v);
    auto [vx, vy, vz] = (v = v / O).val;
    auto cosO = cos(O);
    auto sinO = sin(O);

    mat_t V{0, -vz, vy, vz, 0, -vx, -vy, vx, 0};
    mat_t R = cosO * mat_t::eye() + sinO * V + (Ty_(1) - cosO) * v * v.t();

    return R;
}

template <typename Ty_>
cv::Vec<Ty_, 3> rodrigues(cv::Matx<Ty_, 3, 3> m)
{
    //cv::Vec<Ty_, 3> vec;
    //cv::Rodrigues(m, vec);
    //return vec;

    auto O = acos((cv::trace(m) - (Ty_)1) / (Ty_)2);
    auto v = (Ty_(1) / (Ty_(2) * sin(O))) * cv::Vec<Ty_, 3>(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1));

    return v * O;
}

template <typename Ty_, int r0, int c0, int r1, int c1>
void copyMatx(cv::Matx<Ty_, r0, c0>& to, cv::Matx<Ty_, r1, c1> const& from, int r, int c)
{
    static_assert(r0 >= r1);
    static_assert(c0 >= c1);
    assert(r + r1 <= r0);
    assert(c + c1 <= c0);

    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c1; ++j) {
            to(i + r, j + c) = from(i, j);
        }
    }
}

template <typename Ty_>
void fit_contour_to_screen(std::vector<Ty_>& pts, cv::Rect screen)
{
    // 각 점을 3차원으로 변환합니다. x, y축을 사용
    thread_local static std::vector<cv::Vec3f> vertexes;
    vertexes.clear();
    for (cv::Vec2i pt : pts) {
        vertexes.emplace_back(pt[0], pt[1], 0);
    }

    // 4개의 평면 생성
    auto tl = screen.tl(), br = screen.br();
    plane_t planes[] = {
      {{+1, 0, 0}, -tl.x},
      {{-1, 0, 0}, +br.x},
      {{0, +1, 0}, -tl.y},
      {{0, -1, 0}, +br.y},
    };

    cull_frustum_impl(vertexes, planes, *(&planes + 1) - planes);

    pts.clear();
    for (int it = 0; auto vt : vertexes) {
        pts.emplace_back(cv::Vec2i(vt[0], vt[1]));
    }
}

template <typename Rand_, typename Ty_, int Sz_>
void random_vector(Rand_& rand, cv::Vec<Ty_, Sz_>& vec, Ty_ range)
{
    std::uniform_real_distribution<Ty_> distr{-1, 1};
    vec[0] = distr(rand);
    vec[1] = distr(rand);
    vec[2] = distr(rand);
    vec = cv::normalize(vec) * range;
}
} // namespace billiards::imgproc