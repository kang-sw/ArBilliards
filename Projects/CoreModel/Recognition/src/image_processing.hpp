#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <random>
#include "kangsw/counter.hxx"
#include "kangsw/for_each.hxx"
#include "kangsw/hash_index.hxx"
#include "recognition.hpp"

namespace cv {
template <int Size_, typename Ty_>
void to_json(nlohmann::json& j, const Vec<Ty_, Size_>& v)
{
    j = (std::array<Ty_, Size_>&)v;
}

template <int Size_, typename Ty_>
void from_json(const nlohmann::json& j, Vec<Ty_, Size_>& v)
{
    std::array<Ty_, Size_> const& arr = j;
    v = (cv::Vec<Ty_, Size_>&)arr;
}

template <typename Ty_>
void to_json(nlohmann::json& j, const Scalar_<Ty_>& v)
{
    j = (std::array<Ty_, 4>&)v;
}

template <typename Ty_>
void from_json(const nlohmann::json& j, Scalar_<Ty_>& v)
{
    for (int i = 0, num_elem = min(j.size(), 4ull); i < num_elem; ++i) {
        v.val[i] = j[i];
    }
}
} // namespace cv

namespace std {
template <typename Ty_, size_t N_>
auto begin(cv::Vec<Ty_, N_>& it)
{
    return begin(it.val);
}
template <typename Ty_, size_t N_>
auto end(cv::Vec<Ty_, N_>& it)
{
    return end(it.val);
}
template <typename Ty_, size_t N_>
auto begin(cv::Vec<Ty_, N_> const& it)
{
    return begin(it.val);
}
template <typename Ty_, size_t N_>
auto end(cv::Vec<Ty_, N_> const& it)
{
    return end(it.val);
}

} // namespace std

namespace billiards::imgproc {
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

void range_filter(cv::InputArray input, cv::OutputArray output, cv::Vec3f min_hsv, cv::Vec3f max_hsv);
bool is_border_pixel(cv::Rect img_size, cv::Vec2i pixel, int margin = 3);
void get_table_model(std::vector<cv::Vec3f>& vertexes, cv::Vec2f model_size);
auto get_camera_matx(billiards::recognizer_t::frame_desc const& img) -> std::pair<cv::Matx33d, cv::Matx41d>;
void cull_frustum_impl(std::vector<cv::Vec3f>& obj_pts, plane_t const* plane_ptr, size_t num_planes);
void cull_frustum(std::vector<cv::Vec3f>& obj_pts, std::vector<plane_t> const& planes);
void project_model_local(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes);
void project_points(std::vector<cv::Vec3f> const& points, cv::Matx33f const& camera, cv::Matx41f const& disto, std::vector<cv::Vec2f>& o_points);
auto get_transform_matx_fast(cv::Vec3f pos, cv::Vec3f rot) -> cv::Matx44f;
void transform_points_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes);
void project_model_fast(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, cv::Vec3f obj_pos, cv::Vec3f obj_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes);
auto generate_frustum(float hfov_rad, float vfov_rad) -> std::vector<plane_t>;
void project_model(img_t const& img, std::vector<cv::Vec2f>& mapped_contours, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h = 88, float FOV_v = 50);
void draw_axes(img_t const& img, cv::Mat const& dest, cv::Vec3f rvec, cv::Vec3f tvec, float marker_length, int thickness);
void camera_to_world(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec);
void world_to_camera(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec);
auto rotate_euler(cv::Vec3f target, cv::Vec3f euler_rvec) -> cv::Vec3f;

//
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
void carve_outermost_pixels(cv::InputOutputArray io, cv::Scalar as);
void project_model_points(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes);
void draw_circle(img_t const& img, cv::Mat& dest, float base_size, cv::Vec3f tvec_world, cv::Scalar color, int thickness);
void generate_normalized_sparse_kernel(std::vector<cv::Vec<float, 2>>& normal_random_samples, std::vector<cv::Vec<float, 2>>& normal_negative_samples, cv::Vec2f positive_area_range, cv::Vec2f negative_area_range, int rand_seed, int circle_radius, double rotate_angle);
cv::Point project_single_point(img_t const& img, cv::Vec3f vertex, bool is_world = true);

std::string mat_info_str(cv::Mat const& i);
void color_space_to_flag(kangsw::hash_index cls, int& to, int& from);

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

template <typename Ty_, size_t r0, size_t c0, size_t r1, size_t c1>
void copy_matx(cv::Matx<Ty_, r0, c0>& to, cv::Matx<Ty_, r1, c1> const& from, size_t row_ofst, size_t col_ofst)
{
    static_assert(r0 >= r1);
    static_assert(c0 >= c1);
    assert(row_ofst + r1 <= r0);
    assert(col_ofst + c1 <= c0);

    for (size_t i = 0; i < r1; ++i) {
        for (size_t j = 0; j < c1; ++j) {
            to(i + row_ofst, j + col_ofst) = from(i, j);
        }
    }
}

template <size_t RowOfst_, size_t ColOfst_, size_t W_ = -1, size_t H_ = -1, typename Ty_, size_t SrcW_, size_t SrcH_>
auto submatx(cv::Matx<Ty_, SrcH_, SrcW_> const& src)
{
    constexpr size_t row = RowOfst_;
    constexpr size_t col = ColOfst_;
    constexpr size_t w = W_ == -1 ? SrcW_ - ColOfst_ : W_;
    constexpr size_t h = H_ == -1 ? SrcH_ - RowOfst_ : H_;
    static_assert(row + h <= SrcH_ && col + w <= SrcW_);

    cv::Matx<Ty_, W_, H_> value;
    for (auto r : kangsw::iota(row, row + h)) {
        for (auto c : kangsw::iota(col, col + w)) {
            value(r, c) = src(r + row, c + col);
        }
    }

    return value;
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
    static const std::uniform_real_distribution<Ty_> distr{-1, 1};
    for (int i = 0; i < Sz_; ++i) { vec[i] = distr(rand); }
    vec = cv::normalize(vec) * range;
}

/**
 * 가장 가까운 각각의 컨투어에 대해, 최소 거리에 대해 평가 함수를 적용합니다.
 */
template <typename Fn_>
static float contour_min_dist_for_each(std::vector<cv::Vec2f> const& ct_a, std::vector<cv::Vec2f>& ct_b, Fn_&& eval)
{
    float sum = 0;

    for (auto& pt : ct_a) {
        if (ct_b.empty()) { break; }

        float min_dist = std::numeric_limits<float>::max();
        int min_idx = 0;
        for (int i = 0; i < ct_b.size(); ++i) {
            auto dist = cv::norm(ct_b[i] - pt, cv::NORM_L2SQR);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }

        sum += eval(min_dist);
        ct_b[min_idx] = ct_b.back();
        ct_b.pop_back();
    }

    return sum;
}

template <typename Fn_>
void circle_op(int cent_x, int cent_y, int radius, Fn_&& op)
{
    int x = 0, y = radius;
    int d = 1 - radius;             // 결정변수를 int로 변환
    int delta_e = 3;                // E가 선택됐을 때 증분값
    int delta_se = -2 * radius + 5; // SE가 선탣됐을 때 증분값

    op(cent_x + x, cent_y + y);
    op(cent_x - x, cent_y + y);
    op(cent_x + x, cent_y - y);
    op(cent_x - x, cent_y - y);
    op(cent_x + y, cent_y + x);
    op(cent_x - y, cent_y + x);
    op(cent_x + y, cent_y - x);
    op(cent_x - y, cent_y - x);

    // 12시 방향에서 시작해서 시계방향으로 회전한다고 했을 때
    // 45도를 지나면 y값이 x값보다 작아지는걸 이용
    while (y > x) {
        // E 선택
        if (d < 0) {
            d += delta_e;
            delta_e += 2;
            delta_se += 2;
        }
        // SE 선택
        else {
            d += delta_se;
            delta_e += 2;
            delta_se += 4;
            y--;
        }
        x++;

        op(cent_x + x, cent_y + y);
        op(cent_x - x, cent_y + y);
        op(cent_x + x, cent_y - y);
        op(cent_x - x, cent_y - y);
        op(cent_x + y, cent_y + x);
        op(cent_x - y, cent_y + x);
        op(cent_x + y, cent_y - x);
        op(cent_x - y, cent_y - x);
    }
}

template <typename Fn_>
void circle_op(int radius, Fn_&& op)
{
    int x = 0, y = radius;
    int d = 1 - radius;             // 결정변수를 int로 변환
    int delta_e = 3;                // E가 선택됐을 때 증분값
    int delta_se = -2 * radius + 5; // SE가 선탣됐을 때 증분값

    op(+x, +y), op(-x, +y), op(+x, -y), op(-x, -y);
    op(+y, +x), op(-y, +x), op(+y, -x), op(-y, -x);

    // 12시 방향에서 시작해서 시계방향으로 회전한다고 했을 때
    // 45도를 지나면 y값이 x값보다 작아지는걸 이용
    while (y > x) {
        // E 선택
        if (d < 0) {
            d += delta_e;
            delta_e += 2;
            delta_se += 2;
        }
        // SE 선택
        else {
            d += delta_se;
            delta_e += 2;
            delta_se += 4;
            y--;
        }
        x++;

        op(+x, +y), op(-x, +y), op(+x, -y), op(-x, -y);
        op(+y, +x), op(-y, +x), op(+y, -x), op(-y, -x);
    }
}

template <typename Ty_, typename Rand_>
void discard_random_args(std::vector<Ty_>& iovec, size_t target_size, Rand_&& rengine)
{
    using namespace std;
    while (iovec.size() > target_size) {
        auto ridx = uniform_int_distribution<size_t>{0, iovec.size() - 1}(rengine);
        iovec[ridx] = move(iovec.back());
        iovec.pop_back();
    }
}

template <typename Ty_, size_t I_, size_t J_>
cv::Vec<Ty_, I_ + J_> concat_vec(cv::Vec<Ty_, I_> const& a, cv::Vec<Ty_, I_> const& b)
{
    cv::Vec<Ty_, I_ + J_> r;
    constexpr auto _0_to_i = kangsw::iota{I_};
    constexpr auto _i_to_j = kangsw::iota{I_, J_};
    for (auto i : _0_to_i) { r(i) = a(i); }
    for (auto i : _i_to_j) { r(i) = b(i); }

    return r;
}

template <typename Ty_, size_t I_, typename... Args_>
decltype(auto) concat_vec(cv::Vec<Ty_, I_> const& a, Args_&&... args)
{
    cv::Vec<Ty_, I_ + sizeof...(args)> r;
    constexpr auto _0_to_i = kangsw::iota{I_};
    for (auto i : _0_to_i) { r(i) = a(i); }
    auto tup = std::forward_as_tuple(std::forward<Args_>(args)...);
    kangsw::tuple_for_each(tup, [&](auto&& arg, size_t i) { r(I_ + i) = std::forward<decltype(arg)>(arg); });

    return r;
}

template <typename Ty_, typename... Args_>
decltype(auto) make_vec(Args_&&... args)
{
    cv::Vec<Ty_, sizeof...(args)> r;
    auto tup = std::make_tuple(std::forward<Args_>(args)...);
    kangsw::tuple_for_each(tup, [&](auto&& arg, size_t i) { r(i) = arg; });

    return r;
}

template <size_t Ofst_, size_t Cnt_, typename Ty_, size_t I_>
cv::Vec<Ty_, Cnt_> subvec(cv::Vec<Ty_, I_> const& v)
{
    static_assert(Ofst_ + Cnt_ <= I_);
    return *(cv::Vec<Ty_, Cnt_> const*)(v.val + Ofst_);
}

template <size_t Ofst_, size_t Cnt_, typename Ty_, size_t I_>
cv::Vec<Ty_, Cnt_>& subvec(cv::Vec<Ty_, I_>& v)
{
    static_assert(Ofst_ + Cnt_ <= I_);
    return *(cv::Vec<Ty_, Cnt_>*)(v.val + Ofst_);
}

template <typename Ty_>
cv::Mat_<Ty_> index_by(cv::Mat_<Ty_> const& sources, cv::Mat1i const& indexes)
{
    cv::Mat_<Ty_> retval(indexes.size());
    auto range = kangsw::iota{indexes.size().area()};
    for (auto i : range) { retval(i) = sources(indexes(i)); }
    return retval;
}

} // namespace billiards::imgproc