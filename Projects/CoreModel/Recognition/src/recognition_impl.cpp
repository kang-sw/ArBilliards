#include "recognition_impl.hpp"
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core/base.hpp>
#include <execution>
#include <random>
#include <algorithm>
#include <vector>
#include <nlohmann/json.hpp>
#include "templates.hxx"

using namespace billiards;
using namespace std;

#define YTRACE(x, y) \
    timer_scope_t TM_##x##__##y { this, #x }
#define XTRACE(x, y) YTRACE(x, y)

#define YTRACE_2(x, y, z) \
    timer_scope_t TM_##x##__##y { this, z }
#define XTRACE_2(x, y, z) YTRACE_2(x, y, z)

#define ELAPSE_SCOPE(name) XTRACE_2(TIMER__, __COUNTER__, (name))
#define ELAPSE_BLOCK(name) if (ELAPSE_SCOPE((name)); true)

#define varset(varname)       ((void)billiards::names::varname, vars[#varname])
#define varget(type, varname) ((void)billiards::names::varname, any_cast<remove_reference_t<type>&>(vars[#varname]))

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

template <typename Rand_, typename Ty_, int Sz_>
void random_vector(Rand_& rand, cv::Vec<Ty_, Sz_>& vec, Ty_ range)
{
    uniform_real_distribution<Ty_> distr{-1, 1};
    vec[0] = distr(rand);
    vec[1] = distr(rand);
    vec[2] = distr(rand);
    vec = cv::normalize(vec) * range;
}

template <typename Ty_, typename Rand_>
void discard_random_args(vector<Ty_>& iovec, size_t target_size, Rand_&& rengine)
{
    while (iovec.size() > target_size) {
        auto ridx = uniform_int_distribution<size_t>{0, iovec.size() - 1}(rengine);
        iovec[ridx] = move(iovec.back());
        iovec.pop_back();
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

static int get_pixel_length_on_contact(img_t const& imdesc, plane_t plane, cv::Point pt, float length)
{
    using namespace cv;

    Vec3f far(pt.x, pt.y, 1);
    recognizer_impl_t::get_point_coord_3d(imdesc, far[0], far[1], 1);
    if (auto distance = plane.calc_u({}, far); distance && *distance > 0) {
        auto pxl_len = recognizer_impl_t::get_pixel_length(imdesc, length, *distance);
        return pxl_len;
    }

    return -1;
}

// 주의: 왜곡 고려 안 함!
static void project_points(vector<cv::Vec3f> const& points, cv::Matx33f const& camera, cv::Matx41f const& disto, vector<cv::Vec2f>& o_points)
{
    for (auto& pt : points) {
        auto intm = camera * pt;
        intm /= intm[2];
        o_points.emplace_back(intm[0], intm[1]);
    }
}

static cv::Matx44f get_world_transform_matx_fast(cv::Vec3f pos, cv::Vec3f rot)
{
    using namespace cv;
    Matx44f world_transform = {};
    world_transform(3, 3) = 1.0f;
    {
        world_transform.val[3] = pos[0];
        world_transform.val[7] = pos[1];
        world_transform.val[11] = pos[2];
        Matx33f rot_mat = rodrigues(rot);
        copyMatx(world_transform, rot_mat, 0, 0);
    }
    return world_transform;
}

/**
 * 각 정점에 대해, 시야 사각뿔에 대한 컬링을 수행합니다.
 */
static void cull_frustum_impl(vector<cv::Vec3f>& obj_pts, plane_t const* plane_ptr, size_t num_planes)
{
    using namespace cv;
    // 시야 사각뿔의 4개 평면은 반드시 원점을 지납니다.
    // 평면은 N.dot(P-P1)=0 꼴인데, 평면 상의 임의의 점 P1은 원점으로 설정해 노멀만으로 평면을 나타냅니다.
    auto planes = initializer_list<plane_t>(plane_ptr, plane_ptr + num_planes);
    assert(obj_pts.size() >= 3);

    for (auto pl : planes) {
        auto& o = obj_pts;
        constexpr auto SMALL_NUMBER = 1e-5f;
        // 평면 안의 점 찾기 ... 시작점
        int idx = -1;
        for (int i = 0; i < o.size(); ++i) {
            if (pl.calc(o[i]) >= SMALL_NUMBER) {
                idx = i;
                break;
            }
        }

        // 평면 안에 점 하나도 없으면 드랍
        if (idx == -1) {
            o.clear();
            return;
        }

        for (int nidx, incount = 0; incount < o.size();) {
            nidx = (idx + 1) % o.size();

            // o[idx]는 항상 평면 안

            // o[nidx]도 평면 안에 있다면 스킵
            if (pl.calc(o[nidx]) >= -SMALL_NUMBER) {
                ++incount;
                idx = ++idx % o.size();
                continue;
            }
            incount = 0; // 한 바퀴 더 돌아서 탈출하게 됨

            // o[idx]가 평면 안에 있는 것으로 가정하지만, 애매하게 걸치는 경우 발생
            // 이 경우 o[idx]를 삽입한 점으로 보고, o[nidx]에서 다음 프로세스 시작

            // o[idx]-o[nidx]는 평면을 위에서 아래로 통과
            // 접점 위치에 새로운 점 스폰(o[nidx] 위치)
            // nidx := nidx+1
            if (pl.calc(o[idx]) > 0) {
                auto contact = pl.find_contact(o[idx], o[nidx]).value();
                o.insert(o.begin() + nidx, contact);
                nidx = ++nidx % o.size();
            }

            // o[nidx]에서 출발, 다시 평면 위로 돌아올 때까지 반복
            // A. o[nidx]~o[nidx+1]이 모두 평면 밖에 있다면 o[nidx]는 제거
            // B. 다시 평면 위로 돌아온다면, 평면에 접점 스폰하고 o[nidx]를 대체
            for (int nnidx;;) {
                nnidx = (nidx + 1) % o.size();

                // o[nidx]는 반드시 평면 밖에 있음!
                // o[nnidx]도 평면 밖인 경우 ...
                if (pl.calc(o[nnidx]) < SMALL_NUMBER) {
                    o.erase(o.begin() + nidx);
                    nidx = nidx % o.size(); // index validate
                    continue;
                }

                // 단, o[nidx]가 평면 상에 있는 점일 수 있음
                // 이 경우 이미 o[nidx]를 스폰한 것으로 보고 다음 과정 진행
                if (pl.calc(o[nidx]) < 0) {
                    // o[nnidx]는 평면 안에 있음
                    // 접점을 스폰하고 nidx폐기, 탈출
                    auto contact = pl.find_contact(o[nidx], o[nnidx]).value();
                    o[nidx] = contact;
                }

                idx = nnidx; // nidx는 검증 완료, nnidx에서 새로 시작
                break;
            }
        }
    }
}

template <typename Ty_>
static void fit_contour_to_screen(vector<Ty_>& pts, cv::Rect screen)
{
    // 각 점을 3차원으로 변환합니다. x, y축을 사용
    thread_local static vector<cv::Vec3f> vertexes;
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

static void cull_frustum(vector<cv::Vec3f>& obj_pts, vector<plane_t> const& planes)
{
    cull_frustum_impl(obj_pts, planes.data(), planes.size());
}

static void project_model_local(img_t const& img, vector<cv::Vec2f>& mapped_contour, vector<cv::Vec3f>& model_vertexes, bool do_cull, vector<plane_t> const& planes)
{
    // 오브젝트 포인트에 frustum culling 수행
    if (do_cull) {
        cull_frustum(model_vertexes, planes);
    }

    if (!model_vertexes.empty()) {
        // obj_pts 점을 카메라에 대한 상대 좌표로 치환합니다.
        auto [mat_cam, mat_disto] = recognizer_impl_t::get_camera_matx(img);

        // 각 점을 매핑합니다.
        // projectPoints(model_vertexes, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), mat_cam, mat_disto, mapped_contour);
        project_points(model_vertexes, mat_cam, mat_disto, mapped_contour);
    }
}

static vector<plane_t> generate_frustum(float hfov_rad, float vfov_rad)
{
    using namespace cv;
    vector<plane_t> planes;
    {
        // horizontal 평면 = zx 평면
        // vertical 평면 = yz 평면
        Matx33f rot_vfov;
        Rodrigues(Vec3f(vfov_rad * 0.5f, 0, 0), rot_vfov); // x축 양의 회전
        planes.push_back({rot_vfov * Vec3f{0, 1, 0}, 0});  // 위쪽 면

        Rodrigues(Vec3f(-vfov_rad * 0.53f, 0, 0), rot_vfov);
        // Rodrigues(Vec3f(-vfov_rad * 0.5f, 0, 0), rot_vfov);
        planes.push_back({rot_vfov * Vec3f{0, -1, 0}, 0}); // 아래쪽 면

        Rodrigues(Vec3f(0, hfov_rad * 0.50f, 0), rot_vfov);
        planes.push_back({rot_vfov * Vec3f{-1, 0, 0}, 0}); // 오른쪽 면

        Rodrigues(Vec3f(0, -hfov_rad * 0.508f, 0), rot_vfov);
        //Rodrigues(Vec3f(0, -hfov_rad * 0.5f, 0), rot_vfov);
        planes.push_back({rot_vfov * Vec3f{1, 0, 0}, 0}); // 왼쪽 면
    }

    return move(planes);
}

static void project_model_fast(img_t const& img, vector<cv::Vec2f>& mapped_contour, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull, vector<plane_t> const& planes)
{
    using t = recognizer_impl_t;
    t::transform_to_camera(img, obj_pos, obj_rot, model_vertexes);

    project_model_local(img, mapped_contour, model_vertexes, do_cull, planes);
}

/**
 * 두 컨투어 집합 사이의 거리를 구합니다.
 * 항상 일정한 방향을 가정합니다.
 */
static float contour_distance(vector<cv::Vec2f> const& ct_a, vector<cv::Vec2f>& ct_b)
{
    CV_Assert(ct_a.size() == ct_b.size());

    float sum = 0;

    for (auto& pt : ct_a) {
        float min_dist = numeric_limits<float>::max();
        int min_idx = 0;
        for (int i = 0; i < ct_b.size(); ++i) {
            auto dist = cv::norm(ct_b[i] - pt, cv::NORM_L2SQR);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }

        sum += min_dist;
        ct_b[min_idx] = ct_b.back();
        ct_b.pop_back();
    }

    return sqrt(sum);
}

namespace billiards
{
static int timer_scope_counter;
struct timer_scope_t {
    timer_scope_t(recognizer_impl_t* self, string name)
        : self_(*self)
        , index_(self->elapsed_seconds.size())
    {
        tm_.start();
        auto& arg = self_.elapsed_seconds.emplace_back();
        for (int i = 0; i < timer_scope_counter; i++) {
            arg.first.append("  ");
        }
        arg.first += move(name);
        timer_scope_counter++;
    }

    ~timer_scope_t()
    {
        timer_scope_counter--;
        tm_.stop();
        auto& arg = self_.elapsed_seconds[index_];
        arg.second = chrono::microseconds((int64)tm_.getTimeMicro());
    }

    recognizer_impl_t& self_;
    cv::TickMeter tm_;
    int index_;
};

static bool is_border_pixel(cv::Rect img_size, cv::Vec2i pixel, int margin = 3)
{
    pixel = pixel - (cv::Vec2i)img_size.tl();
    bool w = pixel[0] < margin || pixel[0] >= img_size.width - margin;
    bool h = pixel[1] < margin || pixel[1] >= img_size.height - margin;
    return w || h;
}

optional<recognizer_impl_t::transform_estimation_result_t> recognizer_impl_t::estimate_matching_transform(img_t const& img, vector<cv::Vec2f> const& input_param, vector<cv::Vec3f> model, cv::Vec3f init_pos, cv::Vec3f init_rot, transform_estimation_param_t const& p)
{
    // 입력을 verify합니다. frustum culliing을 활용하기 때문에, 반드시 컨투어 픽셀 두 개 이상이 이미지의 경계선에 걸쳐 있어야 합니다.
    // 적어도 2개 이상의 정점이 이미지 경계 밖에 있어야 하고, 2개 이상의 정점이 경계 안에 있어야 합니다.
    auto input = input_param;
    {
        int num_in = 0, num_out = 0;
        fit_contour_to_screen(input, p.contour_cull_rect);

        // 컬링 된 컨투어 그리기
        vector<cv::Vec2i> pts;
        if (input.size()) {
            pts.assign(input.begin(), input.end());
            cv::drawContours(p.debug_render_mat, vector{{pts}}, -1, {0, 255, 112}, 1);
        }

        for (auto& pt : input) {
            auto is_border = is_border_pixel(p.contour_cull_rect, (cv::Vec2i)pt, p.border_margin);
            num_in += !is_border;
            num_out += is_border;
        }

        // 방향을 구별 가능한 최소 숫자입니다.
        // 또한, 경계에 걸친 점이 적어도 2개 있어야 합니다.
        if (num_in * 2 + num_out < 6 || num_out < 2) {
            return {};
        }
    }

    struct candidate_t {
        float error = numeric_limits<float>::max();
        cv::Vec3f pos, rot;
    };

    using distr_t = uniform_real_distribution<float>;

    distr_t distr_rot_axis{0, p.rot_axis_variant}; // 회전 축의 방향에 다양성 부여

    float pos_variant = p.pos_initial_distance;
    float rot_variant = p.rot_variant;

    vector cands = {candidate_t{numeric_limits<float>::max(), init_pos, init_rot}};

    auto planes = generate_frustum(p.FOV.width * CV_PI / 180.f, p.FOV.height * CV_PI / 180.f);

    bool do_parallel = p.do_parallel;

    for (int iteration = 0; iteration < p.num_iteration; ++iteration) {
        distr_t distr_pos(0, pos_variant);
        distr_t distr_rot(-rot_variant, rot_variant);

        auto pivot_normal = normalize(init_rot);

        if (do_parallel) {
            cands.resize(p.num_candidates);
            thread_local mt19937 rand{(unsigned)hash<thread::id>{}(this_thread::get_id())};
            thread_local vector<cv::Vec2f> ch_mapped; // 메모리 재할당 방지
            thread_local vector<cv::Vec3f> ch_model;  // 메모리 재할당 방지
            for_each(execution::par_unseq, cands.begin(), cands.end(), [&](candidate_t& elem) {
                candidate_t cand;

                // 첫 요소는 항상 기존의 값입니다.
                if (&elem - cands.data() != 0) {
                    auto& p = cand.pos;
                    auto& r = cand.rot;

                    random_vector(rand, p, distr_pos(rand));
                    p += init_pos;

                    // 회전 벡터를 계산합니다.
                    // 회전은 급격하게 변하지 않고, 더 계산하기 까다로우므로 축을 고정하고 회전시킵니다.
                    random_vector(rand, r, distr_rot_axis(rand));
                    r = normalize(r + init_rot); // 회전축에 variant 적용
                    float new_rot_amount = norm(init_rot) + distr_rot(rand);
                    r *= new_rot_amount;
                }
                else {
                    cand = elem;
                }

                // 해당 후보를 화면에 프로젝트
                ch_mapped.clear();
                ch_model.assign(model.begin(), model.end());

                project_model_fast(img, ch_mapped, cand.pos, cand.rot, ch_model, true, planes);
                fit_contour_to_screen(ch_mapped, p.contour_cull_rect);
                // project_model(img, ch_mapped, cand.pos, cand.rot, ch_model, true, p.FOV.width, p.FOV.height);

                // 컨투어 개수가 달라도 기각함에 유의!
                if (ch_mapped.empty() || ch_mapped.size() != input.size()) {
                    cand.error = numeric_limits<float>::max();
                }
                else {
                    float dist_min = contour_distance(input, ch_mapped);
                    cand.error = dist_min;
                }

                elem = cand;
            });
        }
        else {
            vector<cv::Vec2f> ch_mapped; // 메모리 재할당 방지
            vector<cv::Vec3f> ch_model;  // 메모리 재할당 방지
            mt19937 rand(random_device{}());

            while (cands.size() < p.num_candidates) {
                candidate_t cand;
                auto& p = cand.pos;
                auto& r = cand.rot;

                random_vector(rand, p, distr_pos(rand));
                p += init_pos;

                // 회전 벡터를 계산합니다.
                // 회전은 급격하게 변하지 않고, 더 계산하기 까다로우므로 축을 고정하고 회전시킵니다.
                random_vector(rand, r, distr_rot_axis(rand));
                r = normalize(r + init_rot); // 회전축에 variant 적용
                float new_rot_amount = norm(init_rot) + distr_rot(rand);
                r *= new_rot_amount;

                cands.emplace_back(cand);
            }

            // 모든 candidate를 iterate하여 에러를 계산합니다.
            //  #pragma omp parallel for private(ch_mapped) private(ch_model)
            for (int index = 0; index < cands.size(); ++index) {
                auto& cand = cands[index];

                // 해당 후보를 화면에 프로젝트
                ch_mapped.clear();
                ch_model.assign(model.begin(), model.end());

                project_model_fast(img, ch_mapped, cand.pos, cand.rot, ch_model, true, planes);
                // project_model(img, ch_mapped, cand.pos, cand.rot, ch_model, true, p.FOV.width, p.FOV.height);

                // 컨투어 개수가 달라도 기각함에 유의!
                if (ch_mapped.empty() || ch_mapped.size() != input.size()) {
                    cands[index] = cands.back();
                    cands.pop_back();
                    --index; // 인덱스 현재 위치에 유지
                    continue;
                }

                float dist_min = contour_distance(input, ch_mapped);
                cand.error = dist_min;
            }
        }

        // 에러가 가장 적은 candidate를 선택하고, 나머지를 기각합니다.
        if (auto min_it = min_element(cands.begin(), cands.end(), [](auto a, auto b) { return a.error < b.error; });
            min_it != cands.end() && min_it->error < 1e8f) {
            // 탐색 범위를 좁히고 계속합니다.
            pos_variant *= p.iterative_narrow_ratio;
            rot_variant *= p.iterative_narrow_ratio;
            init_pos = min_it->pos;
            init_rot = min_it->rot;

            cands = {*min_it};
        }
        else {
            // 애초에 테이블의 tvec, rvec이 없었던 경우 탐색에 실패할 수 있습니다.
            return {};
        }
    }

    auto& suitable = cands.front();
    transform_estimation_result_t res;
    res.confidence = pow(p.confidence_calc_base, -suitable.error);
    res.position = suitable.pos;
    res.rotation = suitable.rot;

    if (p.render_debug_glyphs && p.debug_render_mat.data) {
        vector<cv::Point> points;
        auto ch_model = model;

        project_model(img, points, res.position, res.rotation, ch_model, true, p.FOV.width, p.FOV.height);
        cv::drawContours(p.debug_render_mat, vector{{points}}, -1, {0, 0, 255}, 1);
    }

    return res;
}

void recognizer_impl_t::project_contours(img_t const& img, const cv::Mat& rgb, vector<cv::Vec3f> model, cv::Vec3f pos, cv::Vec3f rot, cv::Scalar color, int thickness)
{
    vector<cv::Point> mapped;
    cv::Vec2f FOV = m.props["FOV"];
    project_model(img, mapped, pos, rot, model, true, FOV[0], FOV[1]);

    if (!mapped.empty()) {
        drawContours(rgb, vector{{mapped}}, -1, color, thickness);
    }
}

cv::Point recognizer_impl_t::project_single_point(img_t const& img, cv::Vec3f vertex, bool is_world)
{
    using namespace cv;
    vector<Vec3f> pos{{0, 0, 0}};
    vector<Vec2f> pt;

    if (is_world) {
        project_model(img, pt, vertex, {0, 1, 0}, pos, false);
    }
    else {
        project_model_local(img, pt, pos, false, {});
    }

    return static_cast<Vec<int, 2>>(pt.front());
}

void recognizer_impl_t::find_table(img_t const& img, const cv::Mat& debug, const cv::UMat& filtered, vector<cv::Vec2f>& table_contours, nlohmann::json& desc)
{
    using namespace names;
    ELAPSE_SCOPE("Table Search");

    using namespace cv;
    auto p = m.props;
    auto tp = p["table"];
    auto image_size = varget(Size, Size_Image); // any_cast<Size>(vars["scaled-image-size"]);

    {
        ELAPSE_SCOPE("Cotour Approximation & Selection");
        auto tc = tp["contour"];

        vector<vector<Point>> candidates;
        vector<Vec4i> hierarchy;
        findContours(filtered, candidates, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        // 테이블 전체가 시야에 없을 때에도 위치를 추정할 수 있도록, 가장 큰 영역을 기록해둡니다.
        auto max_size_arg = make_pair(-1, 0.0);
        auto eps0 = tc["approx-epsilon-preprocess"];
        auto eps1 = tc["approx-epsilon-convexhull"];
        auto size_threshold = (double)tc["area-threshold-ratio"] * image_size.area();

        // 사각형 컨투어 찾기
        for (int idx = 0; idx < candidates.size(); ++idx) {
            auto& contour = candidates[idx];

            approxPolyDP(vector(contour), contour, eps0, true);
            auto area_size = contourArea(contour);
            if (area_size < size_threshold) {
                continue;
            }

            if (max_size_arg.second < area_size) {
                max_size_arg = {idx, area_size};
            }

            convexHull(vector(contour), contour, true);
            approxPolyDP(vector(contour), contour, eps1, true);
            putText(debug, (stringstream() << "[" << contour.size() << ", " << area_size << "]").str(), contour[0], FONT_HERSHEY_PLAIN, 1.0, {0, 255, 0});

            bool const table_found = contour.size() == 4;

            if (table_found) {
                // marshal
                for (auto& pt : contour) {
                    table_contours.push_back(Vec2f(pt.x, pt.y));
                }
            }
        }

        if (table_contours.empty() && max_size_arg.first >= 0) {
            for (auto& pt : candidates[max_size_arg.first]) {
                table_contours.emplace_back(pt.x, pt.y);
            }

            vector<Vec2i> pts;
            pts.assign(table_contours.begin(), table_contours.end());
            drawContours(debug, vector{{pts}}, -1, {0, 0, 0}, 3);
        }
    }

    // 가장 높은 confidence ...
    bool is_any_border_point = false;
    for (auto pt : table_contours) {
        if (is_border_pixel({{}, image_size}, (Vec2i)pt)) {
            is_any_border_point = true;
            break;
        }
    }
    float confidence = 0.f;

    if (table_contours.size() == 4 && !is_any_border_point) {
        ELAPSE_SCOPE("CASE 1 - PNP Solver");
        vector<Vec3f> obj_pts;
        Vec3d tvec;
        Vec3d rvec;

        get_table_model(obj_pts, tp["size"]["fit"], varget(float, Float_TableOffset));

        // tvec의 초기값을 지정해주기 위해, 깊이 이미지를 이용하여 당구대 중점 위치를 추정합니다.
        bool estimation_valid = true;
        vector<cv::Vec3d> table_points_3d = {};
        {
            // 카메라 파라미터는 컬러 이미지 기준이므로, 깊이 이미지 해상도를 일치시킵니다.
            cv::Mat depth = img.depth;

            auto& c = img.camera;

            // 당구대의 네 점의 좌표를 적절한 3D 좌표로 변환합니다.
            for (auto const& uv : table_contours) {
                auto u = uv[0];
                auto v = uv[1];
                auto z_metric = depth.at<float>(v, u);

                auto& pt = table_points_3d.emplace_back();
                pt[2] = z_metric;
                pt[0] = z_metric * ((u - c.cx) / c.fx);
                pt[1] = z_metric * ((v - c.cy) / c.fy);
            }
        }
        bool solve_successful = false;
        /*
        solve_successful = estimation_valid;
        /*/
        // 3D 테이블 포인트를 바탕으로 2D 포인트를 정렬합니다.
        // 모델 공간에서 테이블의 인덱스는 짧은 쿠션에서 시작해 긴 쿠션으로 반시계 방향 정렬된 상태입니다. 이미지에서 검출된 컨투어는 테이블의 반시계 방향 정렬만을 보장하므로, 모델 공간에서의 정점과 같은 순서가 되도록 contour를 재정렬합니다.
        {
            assert(table_contours.size() == table_points_3d.size());

            // 오차를 감안해 공간에서 변의 길이가 table size의 mean보다 작은 값을 선정합니다.
            auto thres = sum((Vec2f)tp["size"]["fit"])[0] * 0.5;
            for (int idx = 0; idx < table_contours.size() - 1; idx++) {
                auto& t = table_points_3d;
                auto& c = table_contours;
                auto len = norm(t[idx + 1] - t[idx], NORM_L2);

                // 다음 인덱스까지의 거리가 문턱값보다 짧다면 해당 인덱스를 가장 앞으로 당깁니다(재정렬).
                if (len < thres) {
                    c.insert(c.end(), c.begin(), c.begin() + idx);
                    t.insert(t.end(), t.begin(), t.begin() + idx);
                    c.erase(c.begin(), c.begin() + idx);
                    t.erase(t.begin(), t.begin() + idx);

                    break;
                }
            }
            auto [mat_cam, mat_disto] = get_camera_matx(img);

            solve_successful = solvePnP(obj_pts, table_contours, mat_cam, mat_disto, rvec, tvec, false, SOLVEPNP_ITERATIVE);
        }
        //*/

        if (solve_successful) {
            // get_table_model(obj_pts, tp["size"]["fit"], varget(float, Float_TableOffset));
            auto vertexes = obj_pts;
            for (auto& vtx : vertexes) {
                vtx = rodrigues(rvec) * vtx + tvec;
            }

            vector<vector<Vec2i>> contours;
            vector<Vec2f> mapped;
            project_model_local(img, mapped, vertexes, false, {});
            contours.emplace_back().assign(mapped.begin(), mapped.end());

            // 각 점을 비교하여 에러를 계산합니다.
            auto& proj = contours.front();
            double error_sum = 0;
            for (size_t index = 0; index < 4; index++) {
                Vec2f projpt = proj[index];
                error_sum += norm(projpt - table_contours[index], NORM_L2SQR);
            }

            confidence = pow(tp["error-base"], -sqrt(error_sum));

            Vec3f tvec_world = tvec, rvec_world = rvec;
            camera_to_world(img, rvec_world, tvec_world);
            set_filtered_table_rot(rvec_world, confidence, confidence > tp["LPF"]["jump-confidence-threshold"]);
            set_filtered_table_pos(tvec_world, confidence, confidence > tp["LPF"]["jump-confidence-threshold"]);
            // draw_axes(img, (Mat&)rgb, rvec_world, tvec_world, 0.08f, 3);
            {
                drawContours(debug, contours, -1, {255, 123, 0}, 3);
                putText(debug, (stringstream() << "table confidence: " << confidence).str(), {0, 24}, FONT_HERSHEY_PLAIN, 1.0, {255, 255, 255});

                vector<Point> pts;
                get_table_model(vertexes, tp["size"]["fit"], varget(float, Float_TableOffset));
                project_model(varget(img_t, Imgdesc), pts, tvec_world, rvec_world, vertexes, true, 80, 50);

                drawContours(debug, vector{{pts}}, -1, {0, 123, 255}, 3);
            }

            if (confidence < tp["confidence-threshold"]) {
                confidence = 0;
            }
        }
    }

    // 테이블을 찾는데 실패한 경우 iteration method를 활용해 테이블 위치를 추정합니다.
    if (!table_contours.empty() && confidence == 0) {
        ELAPSE_SCOPE("CASE 2 - Iterative Projection");

        vector<Vec3f> model;
        get_table_model(model, m.table.recognition_size, varget(float, Float_TableOffset));

        auto init_pos = table_pos;
        auto init_rot = table_rot;

        auto tpa = tp["partial"];

        int num_iteration = tpa["iteration"];
        int num_candidates = tpa["candidates"];
        float rot_axis_variant = tpa["rot-axis-variant"];
        float rot_variant = tpa["rot-amount-variant"];
        float pos_initial_distance = tpa["pos-variant"];

        int border_margin = tpa["border-margin"];

        vector<Vec2i> points;
        points.assign(table_contours.begin(), table_contours.end());
        drawContours(debug, vector{{points}}, -1, {255, 0, 0}, 3);

        vector<Vec2f> input = table_contours;
        transform_estimation_param_t param = {num_iteration, num_candidates, rot_axis_variant, rot_variant, pos_initial_distance, border_margin};
        Vec2f FOV = p["FOV"];
        param.FOV = {FOV[0], FOV[1]};
        param.debug_render_mat = debug;
        param.render_debug_glyphs = true;
        param.do_parallel = tpa["do-parallel"];
        param.iterative_narrow_ratio = tpa["iteration-narrow-coeff"];

        // contour 컬링 사각형을 계산합니다.
        {
            Vec2d offset = tpa["contour-curll-window"]["offset"];
            Vec2d size = tpa["contour-curll-window"]["size"];
            Vec2i img_size = static_cast<Point>(debug.size());

            Rect r{(Point)(Vec2i)offset.mul(img_size), (Size)(Vec2i)size.mul(img_size)};
            if (get_safe_ROI_rect(debug, r)) {
                param.contour_cull_rect = r;
            }
            else {
                param.contour_cull_rect = Rect{{}, debug.size()};
            }
        }

        auto result = estimate_matching_transform(img, input, model, init_pos, init_rot, param);

        if (result.has_value()) {
            auto& res = *result;
            draw_axes(img, const_cast<cv::Mat&>(debug), res.rotation, res.position, 0.07f, 2);
            set_filtered_table_pos(res.position, res.confidence, false);
            set_filtered_table_rot(res.rotation, res.confidence, false);
            confidence = res.confidence;
            putText(debug, (stringstream() << "partial confidence: " << res.confidence).str(), {0, 24}, FONT_HERSHEY_PLAIN, 1.0, {255, 255, 255});
        }
    }

    if (confidence < tp["confidence-threshold"]) {
        table_contours.clear();
        confidence = 0;
    }

    // 이전 테이블 위치를 렌더
    {
        vector<Vec3f> model;
        auto pos = table_pos;
        auto rot = table_rot;

        get_table_model(model, tp["size"]["inner"]);
        project_contours(img, debug, model, pos, rot, {255, 255, 255}, 1);

        get_table_model(model, tp["size"]["outer"], varget(float, Float_TableOffset));
        project_contours(img, debug, model, pos, rot, {0, 255, 0}, 1);

        draw_axes(img, (Mat&)debug, rot, pos, 0.08f, 3);
    }

    desc["Table"]["Translation"] = table_pos;
    desc["Table"]["Orientation"] = (Vec4f&)table_rot;
    desc["Table"]["Confidence"] = confidence;
}

cv::Vec3f recognizer_impl_t::set_filtered_table_pos(cv::Vec3f new_pos, float confidence, bool allow_jump)
{
    if (!allow_jump || norm(new_pos - table_pos) < m.props["table"]["LPF"]["distance-jump-threshold"]) {
        float alpha = (float)m.props["table"]["LPF"]["position"] * confidence;
        return table_pos = (1 - alpha) * table_pos + alpha * new_pos;
    }
    else {
        return table_pos = new_pos;
    }
}

cv::Vec3f recognizer_impl_t::set_filtered_table_rot(cv::Vec3f new_rot, float confidence, bool allow_jump)
{
    // 180도 회전한 경우, 다시 180도 돌려줍니다.
    if (norm(table_rot - new_rot) > (170.0f) * CV_PI / 180.0f) {
        new_rot = rotate_local(new_rot, {0, (float)CV_PI, 0});
    }

    // 노멀이 위를 향하도록 합니다.
    /*cv::Vec3f up{0, 1, 0};
    up = rodrigues(new_rot) * up;
    if (up[1] > 0) {
        new_rot = rotate_local(new_rot, {0, 0, (float)CV_PI});
    }*/

    if (!allow_jump || norm(new_rot - table_rot) < m.props["table"]["LPF"]["rotation-jump-threshold"]) {
        float alpha = (float)m.props["table"]["LPF"]["rotation"] * confidence;
        return table_rot = (1 - alpha) * table_rot + alpha * new_rot;
    }
    else {
        return table_rot = new_rot;
    }
}

void recognizer_impl_t::get_world_transform_matx(cv::Vec3f pos, cv::Vec3f rot, cv::Mat& world_transform)
{
    world_transform = cv::Mat(4, 4, CV_32FC1);
    world_transform.setTo(0);
    world_transform.at<float>(3, 3) = 1.0f;
    {
        // Vec3f rot = (Vec3f&)desc.table.orientation;
        auto tr_mat = world_transform({0, 3}, {3, 4});
        auto rot_mat = world_transform({0, 3}, {0, 3});
        copyTo(pos, tr_mat, {});
        Rodrigues(rot, rot_mat);
    }
}

void recognizer_impl_t::get_camera_matx(img_t const& img, cv::Mat& mat_cam, cv::Mat& mat_disto)
{
    auto& p = img.camera;
    double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...
    double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
    mat_cam = cv::Mat(3, 3, CV_64FC1, M).clone();
    mat_disto = cv::Mat(4, 1, CV_64FC1, disto).clone();
}

std::pair<cv::Matx33d, cv::Matx41d> recognizer_impl_t::get_camera_matx(img_t const& img)
{
    auto& p = img.camera;
    double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

    double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
    return {cv::Matx33d(M), cv::Matx41d(disto)};
}

void recognizer_impl_t::get_table_model(std::vector<cv::Vec3f>& vertexes, cv::Vec2f model_size, float offset)
{
    vertexes.clear();
    auto [half_x, half_z] = (model_size * 0.5f).val;
    vertexes.assign(
      {
        {-half_x, offset, half_z},
        {-half_x, offset, -half_z},
        {half_x, offset, -half_z},
        {half_x, offset, half_z},
      });
}

bool recognizer_impl_t::get_safe_ROI_rect(cv::Mat const& mat, cv::Rect& roi)
{
    if (roi.x >= mat.cols || roi.y >= mat.rows) {
        goto RETURN_FALSE;
    }

    if (roi.x < 0) {
        roi.width += roi.x;
        roi.x = 0;
    }
    if (roi.y < 0) {
        roi.height += roi.y;
        roi.y = 0;
    }

    if (roi.x + roi.width >= mat.cols) {
        roi.width -= (roi.x + roi.width + 1) - mat.cols;
    }

    if (roi.y + roi.height >= mat.rows) {
        roi.height -= (roi.y + roi.height + 1) - mat.rows;
    }

    if (roi.width <= 0 || roi.height <= 0) {
        goto RETURN_FALSE;
    }

    return true;

RETURN_FALSE:;
    roi.width = roi.height = 0;
    return false;
}

std::optional<cv::Mat> recognizer_impl_t::get_safe_ROI(cv::Mat const& mat, cv::Rect roi)
{
    using namespace cv;

    if (get_safe_ROI_rect(mat, roi)) {
        return mat(roi);
    }

    return {};
}

void recognizer_impl_t::get_point_coord_3d(img_t const& img, float& io_x, float& io_y, float z_metric)
{
    auto& c = img.camera;
    auto u = io_x;
    auto v = io_y;

    io_x = z_metric * ((u - c.cx) / c.fx);
    io_y = z_metric * ((v - c.cy) / c.fy);
}

array<float, 2> recognizer_impl_t::get_uv_from_3d(img_t const& img, cv::Point3f const& coord_3d)
{
    array<float, 2> result;
    auto& [u, v] = result;
    auto& [x, y, z] = coord_3d;
    auto c = img.camera;

    u = (c.fx * x) / z + c.cx;
    v = (c.fy * y) / z + c.cy;

    return result;
}

void recognizer_impl_t::filter_hsv(cv::InputArray input, cv::OutputArray output, cv::Vec3f min_hsv, cv::Vec3f max_hsv)
{
    using namespace cv;
    if (max_hsv[0] < min_hsv[0]) {
        UMat mask, hi, lo, temp;
        auto filt_min = min_hsv, filt_max = max_hsv;
        filt_min[0] = 0, filt_max[0] = 255;

        inRange(input, filt_min, filt_max, mask);
        inRange(input, Scalar(min_hsv[0], 0, 0), Scalar(255, 255, 255), hi);
        inRange(input, Scalar(0, 0, 0), Scalar(max_hsv[0], 255, 255), lo);

        bitwise_or(hi, lo, temp);
        bitwise_and(temp, mask, output);
    }
    else {
        inRange(input, min_hsv, max_hsv, output);
    }
}

void recognizer_impl_t::camera_to_world(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec) const
{
    using namespace cv;
    vector<Vec3f> uvw;
    uvw.emplace_back(0.1f, 0, 0);
    uvw.emplace_back(0, -0.1f, 0);
    uvw.emplace_back(0, 0, 0.1f);
    uvw.emplace_back(0, 0, 0);

    Matx33f rot = rodrigues(rvec);
    // Rodrigues(rvec, rot);

    for (auto& pt : uvw) {
        pt = (rot * pt) + tvec;

        auto pt4 = (Vec4f&)pt;
        pt4[3] = 1.0f, pt4[1] *= -1.0f;
        pt4 = img.camera_transform * pt4;
        pt = (Vec3f&)pt4;
    }

    Matx31f u = normalize(uvw[0] - uvw[3]);
    Matx31f v = normalize(uvw[1] - uvw[3]);
    Matx31f w = normalize(uvw[2] - uvw[3]);
    tvec = uvw[3];

    Matx33f rmat;
    copyMatx(rmat, u, 0, 0);
    copyMatx(rmat, v, 0, 1);
    copyMatx(rmat, w, 0, 2);

    rvec = rodrigues(rmat);
}

cv::Vec3f recognizer_impl_t::rotate_local(cv::Vec3f target, cv::Vec3f rvec)
{
    using namespace cv;
    Matx33f axes = Matx33f::eye();
    Matx33f rotator;
    Rodrigues(target, rotator);
    axes = rotator * axes;

    auto roll = rvec[2] * axes.col(2);
    auto pitch = rvec[0] * axes.col(0);
    auto yaw = rvec[1] * axes.col(1);

    Rodrigues(roll, rotator), axes = rotator * axes;
    Rodrigues(pitch, rotator), axes = rotator * axes;
    Rodrigues(yaw, rotator), axes = rotator * axes;

    Vec3f result;
    Rodrigues(axes, result);
    return result;
}

void recognizer_impl_t::draw_axes(img_t const& img, cv::Mat& dest, cv::Vec3f rvec, cv::Vec3f tvec, float marker_length, int thickness) const
{
    using namespace cv;
    vector<Vec3f> pts;
    pts.assign({{0, 0, 0}, {marker_length, 0, 0}, {0, -marker_length, 0}, {0, 0, marker_length}});

    vector<Vec2f> mapped;
    project_model(img, mapped, tvec, rvec, pts, false);

    pair<int, int> pairs[] = {{0, 1}, {0, 2}, {0, 3}};
    Scalar colors[] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
    for (int i = 0; i < 3; ++i) {
        auto [beg, end] = pairs[i];
        auto color = colors[i];

        Point pt_beg(mapped[beg][0], mapped[beg][1]);
        Point pt_end(mapped[end][0], mapped[end][1]);
        line(dest, pt_beg, pt_end, color, thickness);
    }
}

void recognizer_impl_t::draw_circle(img_t const& img, cv::Mat& dest, float base_size, cv::Vec3f tvec_world, cv::Scalar color, int thickness) const
{
    using namespace cv;
    vector<Vec3f> pos{{0, 0, 0}};
    vector<Vec2f> pt;

    project_model(img, pt, tvec_world, {0, 1, 0}, pos, false);

    float size = get_pixel_length(img, base_size, pos[0][2]);
    if (size > 1) {
        circle(dest, Point(pt[0][0], pt[0][1]), size, color, thickness);
    }
}

plane_t plane_t::from_NP(cv::Vec3f N, cv::Vec3f P)
{
    N = cv::normalize(N);

    plane_t plane;
    plane.N = N;
    plane.d = 0.f;

    // auto u = plane.calc_u(P, P + N).value();
    auto d = plane.calc(P);

    plane.d = -d;
    return plane;
}

plane_t plane_t::from_rp(cv::Vec3f rvec, cv::Vec3f tvec, cv::Vec3f up)
{
    using namespace cv;
    auto P = tvec;
    Matx33f rotator = rodrigues(rvec);
    // Matx33f rotator;
    // Rodrigues(rvec, rotator);
    auto N = rotator * up;
    return plane_t::from_NP(N, P);
}

plane_t& plane_t::transform(cv::Vec3f tvec, cv::Vec3f rvec)
{
    using namespace cv;

    auto P = -N * d;
    Matx33f rotator;
    Rodrigues(rvec, rotator);

    N = rotator * N;
    P = rotator * P + tvec;

    return *this = from_NP(N, P);
}

float plane_t::calc(cv::Vec3f const& pt) const
{
    auto v = N.mul(pt);
    auto res = v[0] + v[1] + v[2] + d;
    return res; //abs(res) < 1e-6f ? 0 : res;
}

bool plane_t::has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    return !!find_contact(P1, P2);
    // return calc(P1) * calc(P2) < 0.f;
}

optional<float> plane_t::calc_u(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    auto P3 = -N * d;

    auto upper = N.dot(P3 - P1);
    auto lower = N.dot(P2 - P1);

    if (abs(lower) > 1e-7f) {
        auto u = upper / lower;

        return u;
    }

    return {};
}

optional<cv::Vec3f> plane_t::find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    if (auto uo = calc_u(P1, P2); uo /*&& calc(P1) * calc(P2) < 0*/) {
        auto u = *uo;

        if (u <= 1.f && u >= 0.f) {
            return P1 + (P2 - P1) * u;
        }
    }
    return {};
}

void recognizer_impl_t::transform_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, vector<cv::Vec3f>& model_vertexes)
{
    // cv::Mat world_transform;
    // get_world_transform_matx(world_pos, world_rot, world_transform);
    auto world_transform = get_world_transform_matx_fast(world_pos, world_rot);
    auto inv_camera_transform = img.camera_transform.inv();

    for (auto& opt : model_vertexes) {
        auto pt = (cv::Vec4f&)opt;
        pt[3] = 1.0f;

        pt = inv_camera_transform * world_transform * pt;

        // 좌표계 변환
        pt[1] *= -1.0f;
        opt = (cv::Vec3f&)pt;
    }
}

void recognizer_impl_t::project_model(img_t const& img, vector<cv::Vec2f>& mapped_contours, cv::Vec3f world_pos, cv::Vec3f world_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h, float FOV_v)
{
    project_model_fast(img, mapped_contours, world_pos, world_rot, model_vertexes, do_cull, generate_frustum(FOV_h * CV_PI / 180.0f, FOV_v * CV_PI / 180.0f));
}

void billiards::recognizer_impl_t::project_model(img_t const& img, vector<cv::Point>& mapped, cv::Vec3f obj_pos, cv::Vec3f obj_rot, vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h, float FOV_v)
{
    vector<cv::Vec2f> mapped_vec;
    project_model(img, mapped_vec, obj_pos, obj_rot, model_vertexes, do_cull, FOV_h, FOV_v);

    mapped.clear();
    for (auto& pt : mapped_vec) {
        mapped.emplace_back((int)pt[0], (int)pt[1]);
    }
}

void recognizer_impl_t::find_ball_center(img_t const& img, vector<cv::Point> const& contours_src, ball_find_parameter_t const& p, ball_find_result_t& r)
{
}

void recognizer_impl_t::carve_outermost_pixels(cv::InputOutputArray io, cv::Scalar as)
{
    if (io.isUMat()) {
        auto mat = io.getUMat();
        mat.row(0).setTo(as);
        mat.col(0).setTo(as);
        mat.row(mat.rows - 1).setTo(as);
        mat.col(mat.cols - 1).setTo(as);
    }
    else if (io.isMat()) {
        auto mat = io.getMat();
        mat.row(0).setTo(as);
        mat.col(0).setTo(as);
        mat.row(mat.rows - 1).setTo(as);
        mat.col(mat.cols - 1).setTo(as);
    }
}

void recognizer_impl_t::async_worker_thread()
{
    while (worker_is_alive) {
        {
            unique_lock<mutex> lck(worker_event_wait_mtx);
            worker_event_wait.wait(lck);
        }

        if (!worker_is_alive) {
            break;
        }

        opt_img_t img;
        img_cb_t on_finish;
        if (read_lock lck(img_cue_mtx); img_cue.has_value()) {
            img = move(*img_cue);
            on_finish = move(img_cue_cb);

            img_cue = {};
            img_cue_cb = {};
        }

        if (img.has_value()) {
            img_show_queue.clear();
            auto desc = proc_img(*img);
            {
                write_lock lock(img_snapshot_mtx);
                img_prev = *img;
            }
            {
                write_lock lock(img_show_mtx);
                img_show = img_show_queue;
            }
            {
                write_lock lock(elapsed_seconds_mtx);
                elapsed_seconds_prev = move(elapsed_seconds);
                elapsed_seconds.clear();
            }
            if (on_finish) { on_finish(*img, desc); }
            prev_desc = desc;
        }
    }
}

void recognizer_impl_t::find_balls(nlohmann::json& desc)
{
    using namespace cv;
    auto& p = m.props;
    auto table_contour = varget(vector<cv::Vec2f>, Var_TableContour);

    // -- 테이블 영역을 Perspective에서 Orthogonal하게 투영합니다.
    // (방법은 아직 연구가 필요 ..)
    // - 이미지는 이미 rectify된 상태이므로, 카메라 파라미터는 따로 고려하지 않음
    // - 테이블의 perspective point로부터 당구대 이미지 획득
    // - 해당 이미지를 테이블을 orthogonal로 투영한 이미지 영역으로 트랜스폼
    // (참고로, Orthogonal하게 투영된 이미지는 원근 X)

    // -- 각 색상의 가장 우수한 후보를 선정
    // - 위에서 Orthogonal Transform을 통해 얻은 이미지 사용
    // - 필드에서 빨강, 오렌지, 흰색 각 색상의 HSV 값을 뺌
    // - 뺀 값 각각에 가중치를 주어 합산(reduce) (H가 가장 크게, V를 가장 작게)
    // - 해당 값의 음수를 pow의 지수로 둠 ... pow(base, -weight); 즉 거리가 멀수록 0에 가까운 값 반환
    // - 고정 커널 크기(Orthogonal로 Transform했으므로 ..)로 컨볼루션 적용, 로컬 맥시멈 추출 ... 공 candidate

    // NOTE: 경계선 평가는 기존의 방법을 사용하되, 이진 이미지가 아닌 HSV 색공간의 Value 표현으로부터 경계선을 검출합니다. Value 표현에 gradient를 먹이고 증폭하면 경계선 이미지를 얻을 수 있을듯? 이후 contour를 사용하는 대신, 원 인덱스 픽셀에 대해 검사하여 가장 높은 가중치를 획득합니다.
    // NOTE: 커널 연산 개선, 정사각형 범위에서 iterate 하되, 반지름 안에 들어가는 픽셀만 가중치를 유효하게 계산합니다.
    // NOTE: 커널 가중치 계산 시 거리가 가까운 픽셀이 우세하므로, 계산된 픽셀 개수로 나눠줍니다.

    // 기존 방법을 근소하게 개선하는 방향으로 ..
    // ROI 획득
    ELAPSE_SCOPE("Ball Tracking");
    auto b = p["ball"];

    auto debug = varget(cv::Mat, Img_Debug);
    cv::UMat u_rgb;
    cv::UMat u_hsv;

    auto ROI = boundingRect(table_contour);
    if (!get_safe_ROI_rect(debug, ROI)) {
        return;
    }
    auto area_mask = varget(cv::Mat, Img_TableAreaMask)(ROI);
    u_rgb = varget(cv::UMat, UImg_RGB)(ROI);
    u_hsv = varget(cv::UMat, UImg_HSV)(ROI);
    show("Ball ROI Area Mask", area_mask);

    vector<cv::UMat> channels;

    split(u_hsv, channels);
    auto [u_h, u_s, u_v] = (cv::UMat(&)[3])(*channels.data());

    // 색상 매칭을 수행합니다.
    // - 공의 기준 색상을 빼고(h, s 채널만), 각 채널에 가중치를 두어 유클리드 거리 d_n를 계산합니다.
    // - base^(-d_n)를 각 픽셀에 대해 계산합니다. (e^(-d_n*ln(base) 로 계산)
    // - 테이블의 가장 가까운 점으로부터 공의 최대 반경을 계산합니다.
    // - 문턱값 이상의 인덱스를 선별합니다(findNoneZero + mask)
    // - 선별된 인덱스와 value 채널로부터 강조된 경계선 이미지를 통해 각각의 점에 대해 경계 적합도를 검사합니다.
    ELAPSE_BLOCK("Ball Candidate Finding")
    {
        auto& bm = b["common"];
        cv::UMat u_match_map[3]; // 공의 각 색상에 대한 매치 맵입니다.
        cv::UMat u0, u1;         // 임시 변수 리스트

        // h, s 채널만 사용합니다.
        // 값 형식은 32F이어야 합니다.
        merge(vector{{u_h, u_s}}, u0);
        u0.convertTo(u1, CV_32FC2, 1 / 255.f);
        for (int i = 0; i < 3; ++i) {
            u_match_map[i] = UMat(ROI.size(), CV_32FC2);
            u_match_map[i].setTo(0);
            u1.copyTo(u_match_map[i], area_mask);
        }

        // 각각의 색상에 대해 매칭을 수행합니다.
        auto depth = (cv::Mat1f&)varget(cv::Mat, Img_Depth);
        auto imdesc = varget(img_t, Imgdesc);
        auto intr__balls__ = {b["red"], b["orange"], b["white"]};
        auto balls = intr__balls__.begin();
        char const* ball_names[] = {"Red", "Orange", "White"};
        float ball_radius = bm["radius"];

        // 테이블 평면 획득
        auto table_plane = plane_t::from_rp(table_rot, table_pos, {0, 1, 0});
        plane_to_camera(imdesc, table_plane, table_plane);

        // 컬러 스케일

        ELAPSE_BLOCK("Matching Field Generation")
        for (int ball_idx = 0; ball_idx < 3; ++ball_idx) {
            auto& m = u_match_map[ball_idx];
            cv::Scalar color = (cv::Vec2f)balls[ball_idx]["color"] / 255;
            cv::Scalar weight = (cv::Vec2f)balls[ball_idx]["weight-hue-sat"];
            weight /= norm(weight);
            auto ln_base = log((double)balls[ball_idx]["error-function-base"]);

            cv::subtract(m, color, u0);
            multiply(u0, u0, u1);
            cv::multiply(u1, weight, u0);

            cv::reduce(u0.reshape(1, u0.rows * u0.cols), u1, 1, cv::REDUCE_SUM);
            u1 = u1.reshape(1, u0.rows);
            sqrt(u1, u0);

            multiply(u0, -ln_base, u1);
            exp(u1, m);

            show("Ball Match Field Raw: "s + ball_names[ball_idx], m);
        }

        cv::UMat match_field{ROI.size(), CV_8UC3};
        match_field.setTo(0);
        cv::Scalar color_ROW[] = {{41, 41, 255}, {0, 213, 255}, {255, 255, 255}};

        // 정규화된 랜덤 샘플의 목록을 만듭니다.
        // 일반적으로 샘플의 아래쪽 반원은 음영에 의해 가려지게 되므로, 위쪽 반원의 샘플을 추출합니다.
        // 이는 정규화된 목록으로, 실제 샘플을 추출할때는 추정 중점 위치에서 계산된 반지름을 곱한 뒤 적용합니다.
        vector<cv::Vec2f> normal_random_samples;
        vector<cv::Vec2f> normal_negative_samples;
        auto& rs = bm["random-sample"];
        ELAPSE_BLOCK("Random Sample Generation")
        {
            mt19937 rg{};
            Vec2f positive_area_range = rs["positive-area"];
            Vec2f negative_area_range = rs["negative-area"];
            for (auto& parea : {&positive_area_range, &negative_area_range}) {
                auto& area = *parea;
                area = area.mul(area);
                if (area[1] < area[0]) {
                    swap(area[1], area[0]);
                }
            }

            uniform_real_distribution<float> distr_positive{positive_area_range[0], positive_area_range[1]};
            uniform_real_distribution<float> distr_negative{negative_area_range[0], negative_area_range[1]};

            if (int rand_seed = (int)rs["seed"]; rand_seed != -1) { rg.seed(rand_seed); }
            int circle_radius = rs["radius"];
            float r0 = -(double)rs["rotate-angle"] * CV_PI / 180;
            cv::Matx22f rotator{cos(r0), -sin(r0), sin(r0), cos(r0)};

            circle_op(0, 0, circle_radius, [&](int xi, int yi) {
                Vec2f vec(xi, yi);
                vec = normalize(vec);

                normal_negative_samples.emplace_back(vec * sqrt(distr_negative(rg)));

                vec[1] = vec[1] > 0 ? -vec[1] : vec[1];
                vec = rotator * vec * sqrt(distr_positive(rg));
                normal_random_samples.emplace_back(vec);
            });

            // 샘플을 시각화합니다.
            int scale = p["others"]["random-sample-view-scale"];
            cv::Mat3b random_sample_visualize(scale, scale);
            random_sample_visualize.setTo(0);
            for (auto& pt : normal_random_samples) {
                random_sample_visualize(cv::Point(pt * scale / 4) + cv::Point{scale / 2, scale / 2}) = {0, 255, 0};
            }
            for (auto& pt : normal_negative_samples) {
                auto at = Point(pt * scale / 4) + cv::Point{scale / 2, scale / 2};
                if (Rect{{}, random_sample_visualize.size()}.contains(at)) {
                    random_sample_visualize(at) = {0, 0, 255};
                }
            }
            show("Random samples", random_sample_visualize);
        }

        cv::Mat1f suitability_field{ROI.size()};
        suitability_field.setTo(0);
        pair<vector<cv::Point>, vector<float>> ball_candidates[3];

        array<Point, 4> ball_positions = {};
        array<float, 4> ball_weights = {};

        ELAPSE_BLOCK("Color/Edge Matching")
        for (int iter = 3; iter >= 0; --iter) {
            auto bidx = max(0, iter - 1); // 0, 1 인덱스는 빨간 공 전용
            auto& m = u_match_map[bidx];
            cv::Mat1f match;

            auto& bp = balls[bidx];
            auto& cand_suitabilities = ball_candidates[bidx].second;
            auto& cand_indexes = ball_candidates[bidx].first;
            int min_pixel_radius = max<int>(1, bm["min-pixel-radius"]);

            ELAPSE_BLOCK("Preprocess: "s + ball_names[bidx])
            {
                m.copyTo(match);

                // color match값이 threshold보다 큰 모든 인덱스를 선택하고, 인덱스 집합을 생성합니다.
                compare(m, (float)bp["suitability-threshold"s], u0, cv::CMP_GT);
                bitwise_and(u0, area_mask, u1);

                // 몇 회의 erode 및 dilate 연산을 통해, 중심에 가까운 픽셀을 골라냅니다.
                dilate(u1, u0, {}, Point(-1, -1), bm["candidate-dilate-count"]);
                erode(u0, u1, {}, Point(-1, -1), bm["candidate-erode-count"]);
                match_field.setTo(color_ROW[bidx], u1);

                // 모든 valid한 인덱스를 추출합니다.
                cand_indexes.reserve(1000);
                findNonZero(u1, cand_indexes);

                // 인덱스를 임의로 골라냅니다.
                auto num_left = rs["sample-max-cases"];
                // size_t num_left = cand_indexes.size() * (100 - clamp(discard, 0, 100)) / 100;
                discard_random_args(cand_indexes, num_left, mt19937{});

                // 매치 맵의 적합도 합산치입니다.
                cand_suitabilities.resize(cand_indexes.size(), 0);
            }
            float negative_weight = balls[bidx]["negative-weight"];

            // 골라낸 인덱스 내에서 색상 값의 샘플 합을 수행합니다.
            ELAPSE_SCOPE("Parallel Launch: "s + ball_names[bidx]);
            auto calculate_suitability = [&](size_t index) {
                auto pt = cand_indexes[index];

                // 현재 추정 위치에서 공의 픽셀 반경 계산
                int ball_pxl_rad = get_pixel_length_on_contact(imdesc, table_plane, pt + ROI.tl(), ball_radius);
                if (ball_pxl_rad < min_pixel_radius) { return; }

                // if 픽셀 반경이 이미지 경계선을 넘어가면 discard
                {
                    cv::Point offset{ball_pxl_rad + 1, ball_pxl_rad + 1};
                    cv::Rect image_bound{offset, ROI.size() - (Size)(offset + offset)};

                    if (!image_bound.contains(pt)) {
                        return;
                    }
                }

                // 각 인덱스에 픽셀 반경을 곱해 매치 맵의 적합도를 합산, 저장
                float suitability = 0;
                {
                    float ball_pxl_radf = ball_pxl_rad;

                    for (auto roundpt : normal_random_samples) {
                        auto sample_index = pt + cv::Point(roundpt * ball_pxl_radf);
                        suitability += match(sample_index);
                    }

                    auto bound = ROI;
                    bound.x = bound.y = 0;
                    for (auto& roundpt : normal_negative_samples) {
                        auto sample_index = pt + Point(roundpt * ball_pxl_radf);
                        if (bound.contains(sample_index)) {
                            suitability -= match(sample_index) * negative_weight;
                        }
                    }
                }

                suitability /= normal_random_samples.size();
                suitability_field(pt) = suitability;
                cand_suitabilities[index] = suitability;
            };

            // 병렬로 launch
            using templates::counter_base;
            if (static_cast<bool>(bm["random-sample"]["do-parallel"])) {
                // for_each(execution::par_unseq, cand_indexes.begin(), cand_indexes.end(), calculate_suitability);
                for_each(execution::par_unseq, counter_base<size_t>{}, counter_base<size_t>{cand_indexes.size()}, calculate_suitability);
            }
            else {
                // for_each(execution::seq, cand_indexes.begin(), cand_indexes.end(), calculate_suitability);
                for_each(execution::seq, counter_base<size_t>{}, counter_base<size_t>{cand_indexes.size()}, calculate_suitability);
            }

            {
                Mat3b debug_ROI = debug(ROI);
                Mat3b adder;
                Mat1b suitability;
                suitability_field.convertTo(suitability, CV_8U, 255);
                //  cv::merge(vector<Mat>{Mat1b::zeros(ROI.size()), suitability, Mat1b::zeros(ROI.size())}, adder);
                cv::merge(vector{suitability, suitability, suitability}, adder);
                debug_ROI -= adder;
            }

            // 특수: 색상이 RED라면 마스크에서 찾아낸 볼에 해당하는 위치를 모두 지우고 위 과정을 다시 수행합니다.
            auto best = max_element(cand_suitabilities.begin(), cand_suitabilities.end());
            if (best == cand_suitabilities.end()) {
                continue;
            }

            if (*best > (float)bm["confidence-threshold"]) {
                auto best_idx = best - cand_suitabilities.begin();
                auto center = cand_indexes[best_idx];

                auto rad_px = get_pixel_length_on_contact(imdesc, table_plane, center + ROI.tl(), ball_radius);

                if (rad_px > 0) {
                    // 공이 정상적으로 찾아진 경우에만 ...
                    // circle(debug, center + ROI.tl(), rad_px, color_ROW[bidx], 1);

                    ball_positions[iter] = center;
                    ball_weights[iter] = *best;

                    // 빨간 공인 경우 ...
                    if (iter == 1) {
                        // Match map에서 검출된 공 위치를 지우고, 위 과정을 반복합니다.
                        circle(m, center, rad_px + (int)balls[0]["second-ball-erase-additional-radius"], 0, -1);
                        show("Ball Match Field Raw: Red 2", m);
                    }
                }
            }
        }

        show("Ball Match Suitability Field"s, suitability_field);
        show("Ball Match Field Mask"s, match_field);
        for (auto& v : ball_weights) { v *= (float)bm["confidence-weight"]; }

        // 각 공의 위치를 월드 기준으로 변환합니다.
        array<Vec3f, 4> ballpos;
        for (int i = 0; i < 4; ++i) {
            if (ball_weights[i] == 0) {
                continue;
            }
            auto pt = ball_positions[i] + ROI.tl();
            Vec3f dst;

            // 공의 중점 방향으로 광선을 투사해 공의 카메라 기준 위치 획득
            dst[0] = pt.x, dst[1] = pt.y, dst[2] = 10;
            get_point_coord_3d(imdesc, dst[0], dst[1], dst[2]);
            auto contact = table_plane.find_contact({}, dst).value();

            Vec3f dummy = {0, 1, 0};
            camera_to_world(imdesc, dummy, contact);
            ballpos[i] = contact;
        }

        // 공의 위치를 이전과 비교합니다.
        struct ball_position_desc_t {
            Vec3f pos;
            Vec3f vel;
            chrono::system_clock::time_point tp;
            double dt(chrono::system_clock::time_point now) const { return chrono::duration<double, chrono::system_clock::period>(now - tp).count(); }
            Vec3f ps(chrono::system_clock::time_point now) const { return dt(now) * vel + pos; }
        };

        using ball_desc_set_t = array<ball_position_desc_t, 4>;
        ball_desc_set_t descs;

        if (varset(Var_PrevBallPos).has_value()) {
            auto prev = varget(ball_desc_set_t, Var_PrevBallPos);
            auto now = chrono::system_clock::now();
            double max_error_speed = b["classification"]["max-error-speed"];

            // 만약 0번 공의 weight가 0인 경우, 즉 공이 하나만 감지된 경우
            // 1번 공의 감지된 위치와 캐시된 0, 1번 공 위치를 비교하고, 1번 공과 더 동떨어진 것을 선택합니다.
            if (ball_weights[0] == 0 && ball_weights[1]) {
                auto p1 = ballpos[1];
                auto diffs = {norm(p1 - prev[0].pos), norm(p1 - prev[1].pos)};
                auto farther = distance(diffs.begin(), max_element(diffs.begin(), diffs.end()));
                ball_weights[0] = 0.51f; // magic number ...
                ballpos[0] = prev[farther].pos;
            }

            // 이전 위치와 비교해, 자리가 바뀐 경우를 처리합니다.
            if (ball_weights[1] && ball_weights[0]) {
                auto p = ballpos[0],
                     ps0 = prev[0].ps(now),
                     ps1 = prev[1].ps(now);

                if (norm(ps1 - p) < norm(ps0 - p)) {
                    swap(ballpos[0], ballpos[1]);
                    swap(ball_weights[0], ball_weights[1]);
                }
            }

            double alpha = bm["movement"]["position-LPF-alpha"];
            double jump_dist = bm["movement"]["jump-distance"];
            for (int i = 0; i < 4; ++i) {
                auto& d = prev[i];

                if (ball_weights[i] < bm["confidence-threshold"]) {
                    continue;
                }

                auto dt = d.dt(now);
                auto dp = ballpos[i] - d.pos;
                auto vel_elapsed = dp / dt;

                // 속도 차이가 오차 범위 이내일때만 이를 반영합니다.
                // 속도가 오차 범위를 벗어난 경우 현재 위치와 속도를 갱신하지 않습니다.
                //
                if (norm(vel_elapsed - d.vel) < max_error_speed) {
                    // 만약 jump distance보다 위치 변화가 적다면, LPF로 위치를 누적합니다.
                    if (norm(dp) < jump_dist) {
                        ballpos[i] = d.pos + (ballpos[i] - d.pos) * alpha;
                    }

                    descs[i] = ball_position_desc_t{.pos = ballpos[i], .vel = vel_elapsed, .tp = now};
                }
                else {
                    ball_weights[i] = 0;
                }
            }
        }
        else {
            for (int i = 0; auto& d : descs) {
                auto pos = ballpos[i++];
                d = ball_position_desc_t{.pos = pos, .vel = {}, .tp = chrono::system_clock::now()};
            }
        }
        varset(Var_PrevBallPos) = descs;
        char const* ball_name_dst[] = {"Red1", "Red2", "Orange", "White"};
        for (int i = 0; i < 4; ++i) {
            auto& dst = desc[ball_name_dst[i]];
            dst["Position"] = descs[i].pos;
            dst["Confidence"] = ball_weights[i];

            // 화면에 디버그용 그림을 그립니다.
            if (ball_weights[i]) {
                draw_circle(imdesc, debug, ball_radius, descs[i].pos, color_ROW[max(i - 1, 0)], 1);
                auto center = project_single_point(imdesc, descs[i].pos);
                auto pxl_rad = get_pixel_length_on_contact(imdesc, table_plane, center + ROI.tl(), ball_radius);

                putText(debug, to_string(i + 1), center + Point(-7, -pxl_rad), FONT_HERSHEY_PLAIN, 1.3, {0, 0, 0}, 2);
            }
        }

        // 테이블에 대한 상대 좌표로 변환
        {
            Vec2f table = p["table"]["size"]["inner"];
            float scale = p["others"]["top-view-scale"];

            int ball_rad_pxl = ball_radius * scale;
            Size table_topview_size = (Point)(Vec2i)(table * scale);

            Mat3b top_view_mat(table_topview_size);
            top_view_mat.setTo(Scalar{243, 16, 37});

            auto inv_tr = get_world_transform_matx_fast(table_pos, table_rot).inv();
            for (int iter = 0; auto& b : descs) {
                int index = iter++;
                int bidx = max(0, index - 1);

                if (ball_weights[index] == 0) {
                    continue;
                }

                Vec4f pos4 = (Vec4f&)b.pos;
                pos4(3) = 1;

                pos4 = inv_tr * pos4;
                auto pt = Point(pos4[0] * scale, -pos4[2] * scale) + (Point)table_topview_size / 2;

                circle(top_view_mat, pt, ball_rad_pxl, color_ROW[bidx], -1);
                putText(top_view_mat, to_string(iter), pt + Point(-6, 11), FONT_HERSHEY_PLAIN, scale * 0.002, {0, 0, 0}, 2);
            }

            show("Table Top Projection", top_view_mat);
        }
    }
}

nlohmann::json recognizer_impl_t::proc_img(img_t const& imdesc_source)
{
    using namespace names;
    using namespace names;
    using namespace cv;
    auto const& p = m.props;
    nlohmann::json desc = {};

    // -- 시작 파라미터 전송
    desc["Table"]["InnerWidth"] = p["table"]["size"]["inner"][0];
    desc["Table"]["InnerHeight"] = p["table"]["size"]["inner"][1];
    desc["BallRadius"] = p["ball"]["common"]["radius"];

    // 깊이를 잘라내기 위한
    {
        array<Vec2f, 2> tf = p["table"]["filter"];
        auto [min, max] = tf;
        enum { H,
               S,
               V };

        desc["Table"]["EnableShaderApplyDepthOverride"] = p["unity"]["enable-table-depth-override"];
        desc["Table"]["ShaderMinH"] = min[H] * (1 / 180.f);
        desc["Table"]["ShaderMaxH"] = max[H] * (1 / 180.f);
        desc["Table"]["ShaderMinS"] = min[S] * (1 / 255.f);
        desc["Table"]["ShaderMaxS"] = max[S] * (1 / 255.f);
    }

    show("Source image", imdesc_source.rgba);

    if (p["__enable"] == false) {
        return desc;
    }

    ELAPSE_SCOPE("TOTAL");

    // 각종 파라미터 등 계산
    {
        // // 테이블 오프셋 계산 ...
        // float height = p["table"]["cushion-height"];
        // float radius = p["ball"]["common"]["radius"];
        varset(Float_TableOffset) = 0.f; // -(height - radius);
    }

    {
        ELAPSE_SCOPE("Image Preprocessing");
        img_t imdesc_scaled = imdesc_source;

        // RGBA 이미지를 RGB로 컨버트합니다.
        Mat img_rgb;
        {
            ELAPSE_SCOPE("RGBA to RGB");

            cvtColor(imdesc_source.rgba, img_rgb, COLOR_RGBA2RGB);
            varset(Img_SrcRGB) = img_rgb;
        }

        // 공용 머터리얼 셋업 시퀀스
        Size scaled_image_size;

        Mat img_rgb_scaled;
        UMat uimg_rgb_scaled;
        if ((bool)p["do-resize"]) {
            ELAPSE_SCOPE("RGB Downsampling");
            scaled_image_size = Size((int)p["fast-process-width"], 0);
            float image_scale = scaled_image_size.width / (float)img_rgb.cols;
            scaled_image_size.height = (int)(img_rgb.rows * image_scale);
            varset(Size_Image) = scaled_image_size;

            // 스케일된 이미지 준비
            resize(img_rgb.getUMat(ACCESS_FAST), uimg_rgb_scaled, scaled_image_size, 0, 0, INTER_LINEAR);
            uimg_rgb_scaled.copyTo(img_rgb_scaled);

            // 스케일된 이미지의 파라미터 준비
            auto scp = imdesc_source.camera;
            for (auto& value : {&scp.fx, &scp.fy, &scp.cx, &scp.cy}) {
                *value *= image_scale;
            }
            imdesc_scaled.camera = scp;
            cvtColor(img_rgb_scaled, imdesc_scaled.rgba, COLOR_RGB2RGBA);
        }
        else {
            varset(Size_Image) = scaled_image_size = img_rgb.size();

            img_rgb_scaled = img_rgb;
            img_rgb_scaled.copyTo(uimg_rgb_scaled);
        }

        Mat img_hsv_scaled;
        UMat uimg_hsv_scaled;

        // 색공간 변환 수행
        {
            ELAPSE_SCOPE("RGB to HSV Conversion");
            varset(Img_RGB) = img_rgb_scaled;
            varset(UImg_RGB) = uimg_rgb_scaled;

            cvtColor(uimg_rgb_scaled, uimg_hsv_scaled, COLOR_RGB2HSV);
            uimg_hsv_scaled.copyTo(img_hsv_scaled);

            varset(Img_HSV) = img_hsv_scaled;
            varset(UImg_HSV) = uimg_hsv_scaled;

            // UMat uimg_ycbcr_scaled;
            // Mat img_ycbcr_scaled;
            // cvtColor(uimg_rgb_scaled, uimg_ycbcr_scaled, COLOR_RGB2HSV);
            // uimg_ycbcr_scaled.copyTo(img_ycbcr_scaled);
            //
            // varset(Img_YCbCr) = img_ycbcr_scaled;
            // varset(UImg_YCbCr) = uimg_ycbcr_scaled;
        }

        // 깊이 이미지 크기 변환
        {
            ELAPSE_SCOPE("Depth Mat resizing");
            UMat u_depth;
            Mat depth;
            resize(imdesc_scaled.depth, u_depth, scaled_image_size);
            u_depth.copyTo(depth);
            varset(UImg_Depth) = u_depth;
            varset(Img_Depth) = depth;

            imdesc_scaled.depth = depth;
        }

        varset(Imgdesc) = imdesc_scaled;
    }

    // 디버깅용 이미지 설정
    varset(Img_Debug) = varget(Mat, Img_RGB).clone();

    // 테이블 탐색을 위한 로직입니다.
    {
        ELAPSE_SCOPE("Table Tracking");
        auto& tp = p["table"];
        auto& u_hsv = varget(UMat, UImg_HSV);
        auto image_size = varget(Size, Size_Image);

        // -- 테이블 색상으로 필터링 수행
        UMat u_mask;
        UMat u_eroded, u_edge;
        {
            ELAPSE_SCOPE("Table Felt Filtering");
            array<Vec3f, 2> filters = tp["filter"];
            filter_hsv(u_hsv, u_mask, filters[0], filters[1]);
            carve_outermost_pixels(u_mask, {0});
            varset(UImg_TableFiltered) = u_mask;
            show("Table Blue Color Mask", u_mask);

            if (
              int prev_iter = tp["preprocess"]["dilate-erode-num-erode-prev"],
              post_iter = tp["preprocess"]["dilate-erode-num-erode-post"];
              prev_iter > 0 && post_iter > 0) {
                ELAPSE_SCOPE("Dilate-Erode Noise Remove");
                UMat u0, u1;
                auto num_dilate = prev_iter + post_iter;
                copyMakeBorder(u_mask, u0, num_dilate, num_dilate, num_dilate, num_dilate, BORDER_CONSTANT);
                erode(u0, u1, {}, {-1, -1}, prev_iter, BORDER_CONSTANT, {});
                dilate(u1, u0, {}, {-1, -1}, num_dilate, BORDER_CONSTANT, {});
                erode(u0, u1, {}, {-1, -1}, post_iter, BORDER_CONSTANT, {});
                u_mask = u1(Rect{{num_dilate, num_dilate}, u_mask.size()});
                show("Table Blue Color Mask - Eroded", u_mask);
            }

            // -- 경계선 검출
            erode(u_mask, u_eroded, {});
            subtract(u_mask, u_eroded, u_edge);
        }

        // -- 테이블 추정 영역 찾기
        vector<Vec2f> table_contour;
        find_table(varget(img_t, Imgdesc), varget(Mat, Img_Debug), varget(UMat, UImg_TableFiltered), table_contour, desc);

        varset(Var_TableContour) = table_contour;
    }

    if (auto& table_contour = varget(vector<Vec2f>, Var_TableContour); table_contour.empty()) {
        vector<Vec3f> model;
        get_table_model(model, p["table"]["size"]["fit"], varget(float, Float_TableOffset));
        project_model(varget(img_t, Imgdesc), table_contour, table_pos, table_rot, model, true, p["FOV"][0], p["FOV"][1]);

        for (auto pt : table_contour) {
            if (isnan(pt[0]) || isnan(pt[1])) {
                table_contour.clear();
                break;
            }
        }
    }

    if (auto& table_contour = varget(vector<Vec2f>, Var_TableContour); !table_contour.empty()) {
        // 컨투어 마스크를 그립니다.
        Mat1b table_area(varget(Size, Size_Image));
        table_area.setTo(0);
        vector<Vec2i> pts;
        pts.assign(table_contour.begin(), table_contour.end());
        drawContours(table_area, vector{{pts}}, -1, 255, -1);
        varset(Img_TableAreaMask) = (Mat)table_area;

        // 테이블 영역의 화이트 밸런스를 조절합니다.
        if (0) {
            ELAPSE_SCOPE("Calculate Histogram");
            vector<Point> indexes;
            findNonZero(table_area, indexes);

            // RGB 히스토그램에서 가장 바깥쪽 픽셀을 드랍합니다.
            Vec3d discard_rgb = p["table"]["preprocess"]["AWB-RGB-discard-rate"];
            Vec3i discard_count = discard_rgb * (double)indexes.size() * 0.5;
            array<int, 256> histo[3] = {};
            auto img = (Mat3b&)varget(Mat, Img_RGB);

            enum : int { R,
                         G,
                         B };

            // Generate Histogram
            for (auto index : indexes) {
                auto color = img(index);
                ++histo[R][color[R]];
                ++histo[G][color[G]];
                ++histo[B][color[B]];
            }

            // 각각의 채널에 대해 버릴 픽셀 인덱스 계산
            Vec2i lo_hi_pivot[3] = {};
            Vec3f mults;
            Vec3f adds;
            for (int C : {R, G, B}) {
                auto& pvt = lo_hi_pivot[C];
                pvt[0] = 0, pvt[1] = 256;
                for (int sum = 0; pvt[0] < 128 && sum < discard_count[C]; sum += histo[C][pvt[0]++]) { }
                for (int sum = 0; pvt[1] > 128 && sum < discard_count[C]; sum += histo[C][--pvt[1]]) { }

                adds[C] = pvt[0];
                mults[C] = 255 / (float)(pvt[1] - pvt[0]);
            }

            for (auto index : indexes) {
                Vec3f pxl = img(index);
                pxl = (pxl - adds).mul(mults);
                img(index) = pxl;
            }
        }

        find_balls(desc);
    }

    // Debugging glyphs
    {
        // 카메라 트랜스폼 그리기
        Vec3f rot;
        Rodrigues(Mat(imdesc_source.camera_transform)({0, 3}, {0, 3}), rot);

        auto imdesc = varget(img_t, Imgdesc);
        auto debug = varget(Mat, Img_Debug);
        draw_axes(imdesc, debug, rot, table_pos + Vec3f{0, 0.3f, 0}, 0.1f, 8);
    }

    show("AAA_RecognitionState", varget(Mat, Img_Debug));

    // ShowImage에 모든 임시 매트릭스 추가
    //ELAPSE_BLOCK("Debugging Mat Copy")
    //for (auto& pair : vars) {
    //    auto& value = pair.second;

    //    if (auto ptr = any_cast<Mat>(&value)) {
    //        show("~Auto: " + move(pair.first), *ptr);
    //    }
    //    else if (auto uptr = any_cast<UMat>(&value)) {
    //        // show(move(pair.first), *uptr);
    //    }
    //}

    return desc;
} // namespace billiards

void recognizer_impl_t::plane_to_camera(img_t const& img, plane_t const& table_plane, plane_t& table_plane_camera)
{
    cv::Vec4f N = (cv::Vec4f&)table_plane.N;
    cv::Vec4f P = -table_plane.d * N;
    N[3] = 0.f, P[3] = 1.f;

    cv::Matx44f camera_inv = img.camera_transform.inv();
    N = camera_inv * N;
    P = camera_inv * P;

    // CV 좌표계로 변환
    N[1] *= -1.f, P[1] *= -1.f;

    table_plane_camera = plane_t::from_NP((cv::Vec3f&)N, (cv::Vec3f&)P);
}

float recognizer_impl_t::get_pixel_length(img_t const& img, float len_metric, float Z_metric)
{
    using namespace cv;

    auto [u1, v1] = get_uv_from_3d(img, Vec3f(0, 0, Z_metric));
    auto [u2, v2] = get_uv_from_3d(img, Vec3f(len_metric, 0, Z_metric));

    return u2 - u1;
}

recognizer_t::recognizer_t()
    : impl_(make_unique<recognizer_impl_t>(*this))
{
}

recognizer_t::~recognizer_t() = default;

void recognizer_t::initialize()
{
    if (!impl_) {
        impl_ = make_unique<recognizer_impl_t>(*this);
    }
}

void recognizer_t::destroy()
{
    impl_.reset();
}

void recognizer_t::refresh_image(parameter_type image, recognizer_t::process_finish_callback_type&& callback)
{
    if (!impl_) {
        return;
    }

    auto& m = *impl_;
    bool img_swap_before_prev_img_proc = false;

    if (write_lock lock(m.img_cue_mtx); lock) {
        img_swap_before_prev_img_proc = !!m.img_cue;
        m.img_cue = image;
        m.img_cue_cb = move(callback);
    }

    if (img_swap_before_prev_img_proc) {
        cout << "warning: image request cued before previous image processed\n";
    }

    m.worker_event_wait.notify_all();
}

void recognizer_t::poll(std::unordered_map<std::string, cv::Mat>& shows)
{
    // 비동기적으로 수집된 이미지 목록을 획득합니다.
    auto& m = *impl_;

    if (read_lock lock(m.img_show_mtx, try_to_lock); lock) {
        for (auto& pair : m.img_show) {
            shows[pair.first] = pair.second;
        }
    }
}

recognizer_t::parameter_type recognizer_t::get_image_snapshot() const
{
    read_lock lock(impl_->img_snapshot_mtx);
    return impl_->img_prev;
}

std::vector<std::pair<std::string, std::chrono::microseconds>> recognizer_t::get_latest_timings() const
{
    read_lock lock{impl_->elapsed_seconds_mtx};
    return impl_->elapsed_seconds_prev;
}
} // namespace billiards

namespace std
{
ostream& operator<<(ostream& strm, billiards::recognizer_t::parameter_type const& desc)
{
    auto write = [&strm](auto val) {
        strm.write((char*)&val, sizeof val);
    };
    write(desc.camera);
    write(desc.camera_transform);
    write(desc.camera_translation);
    write(desc.camera_orientation);

    auto rgba = desc.rgba.clone();
    auto depth = desc.depth.clone();

    write(rgba.rows);
    write(rgba.cols);
    write(rgba.type());
    write((size_t)rgba.total() * rgba.elemSize());

    write(depth.rows);
    write(depth.cols);
    write(depth.type());
    write((size_t)depth.total() * depth.elemSize());

    strm.write((char*)rgba.data, rgba.total() * rgba.elemSize());
    strm.write((char*)depth.data, depth.total() * depth.elemSize());

    return strm;
}

istream& operator>>(istream& strm, billiards::recognizer_t::parameter_type& desc)
{
    auto read = [&strm](auto& val) {
        strm.read((char*)&val, sizeof(remove_reference_t<decltype(val)>));
    };

    read(desc.camera);
    read(desc.camera_transform);
    read(desc.camera_translation);
    read(desc.camera_orientation);

    int rgba_rows, rgba_cols, rgba_type;
    size_t rgba_bytes;
    int depth_rows, depth_cols, depth_type;
    size_t depth_bytes;

    read(rgba_rows);
    read(rgba_cols);
    read(rgba_type);
    read(rgba_bytes);

    read(depth_rows);
    read(depth_cols);
    read(depth_type);
    read(depth_bytes);

    auto& rgba = desc.rgba;
    rgba = cv::Mat(rgba_rows, rgba_cols, rgba_type);
    strm.read((char*)rgba.data, rgba_bytes);

    auto& depth = desc.depth;
    depth = cv::Mat(depth_rows, depth_cols, depth_type);
    strm.read((char*)depth.data, depth_bytes);

    return strm;
}
} // namespace std