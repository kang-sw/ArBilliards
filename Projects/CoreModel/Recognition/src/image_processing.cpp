#include "image_processing.hpp"
#include <execution>

#include "kangsw/misc.hxx"

void billiards::imgproc::cull_frustum_impl(std::vector<cv::Vec3f>& obj_pts, plane_t const* plane_ptr, size_t num_planes)
{
    using namespace cv;
    // 시야 사각뿔의 4개 평면은 반드시 원점을 지납니다.
    // 평면은 N.dot(P-P1)=0 꼴인데, 평면 상의 임의의 점 P1은 원점으로 설정해 노멀만으로 평면을 나타냅니다.
    auto planes = std::initializer_list<plane_t>(plane_ptr, plane_ptr + num_planes);
    assert(obj_pts.size() >= 3);
    auto& o = obj_pts;

    for (auto pl : planes) {
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

void billiards::imgproc::cull_frustum(std::vector<cv::Vec3f>& obj_pts, std::vector<plane_t> const& planes)
{
    cull_frustum_impl(obj_pts, planes.data(), planes.size());
}

void billiards::imgproc::project_model_local(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes)
{
    // 오브젝트 포인트에 frustum culling 수행
    if (do_cull) {
        cull_frustum(model_vertexes, planes);
    }

    if (!model_vertexes.empty()) {
        // obj_pts 점을 카메라에 대한 상대 좌표로 치환합니다.
        auto [mat_cam, mat_disto] = get_camera_matx(img);

        // 각 점을 매핑합니다.
        // projectPoints(model_vertexes, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), mat_cam, mat_disto, mapped_contour);
        project_points(model_vertexes, mat_cam, mat_disto, mapped_contour);
    }
}

void billiards::imgproc::project_points(std::vector<cv::Vec3f> const& points, cv::Matx33f const& camera, cv::Matx41f const& disto, std::vector<cv::Vec2f>& o_points)
{
    for (auto& pt : points) {
        auto intm = camera * pt;
        intm /= intm[2];
        o_points.emplace_back(intm[0], intm[1]);
    }
}

cv::Matx44f billiards::imgproc::get_world_transform_matx_fast(cv::Vec3f pos, cv::Vec3f rot)
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

void billiards::imgproc::transform_to_camera(img_t const& img, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes)
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

void billiards::imgproc::project_model_fast(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, cv::Vec3f obj_pos, cv::Vec3f obj_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes)
{
    transform_to_camera(img, obj_pos, obj_rot, model_vertexes);

    project_model_local(img, mapped_contour, model_vertexes, do_cull, planes);
}

std::vector<billiards::imgproc::plane_t> billiards::imgproc::generate_frustum(float hfov_rad, float vfov_rad)
{
    using namespace cv;
    using namespace std;
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

void billiards::imgproc::project_model(img_t const& img, std::vector<cv::Vec2f>& mapped_contours, cv::Vec3f world_pos, cv::Vec3f world_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h, float FOV_v)
{
    project_model_fast(img, mapped_contours, world_pos, world_rot, model_vertexes, do_cull, generate_frustum(FOV_h * CV_PI / 180.0f, FOV_v * CV_PI / 180.0f));
}

void billiards::imgproc::draw_axes(img_t const& img, cv::Mat const& dest, cv::Vec3f rvec, cv::Vec3f tvec, float marker_length, int thickness)
{
    using namespace cv;
    using namespace std;
    vector<Vec3f> pts;
    pts.assign({{0, 0, 0}, {marker_length, 0, 0}, {0, -marker_length, 0}, {0, 0, marker_length}});

    vector<Vec2f> mapped;
    project_model(img, mapped, tvec, rvec, pts, false);

    pair<int, int> pairs[] = {{0, 1}, {0, 2}, {0, 3}};
    Scalar colors[] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
    for (int i = 0; i < 3; ++i) {
        auto [beg, end] = pairs[i];
        auto color = colors[i];

        Point pt_beg((int)mapped[beg][0], (int)mapped[beg][1]);
        Point pt_end((int)mapped[end][0], (int)mapped[end][1]);
        line(dest, pt_beg, pt_end, color, thickness);
    }
}

void billiards::imgproc::camera_to_world(img_t const& img, cv::Vec3f& rvec, cv::Vec3f& tvec)
{
    using namespace cv;
    std::vector<Vec3f> uvw;
    uvw.emplace_back(0.1f, 0, 0);
    uvw.emplace_back(0, -0.1f, 0);
    uvw.emplace_back(0, 0, 0.1f);
    uvw.emplace_back(0, 0, 0);

    Matx33f rot = rodrigues(rvec);

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

cv::Vec3f billiards::imgproc::rotate_local(cv::Vec3f target, cv::Vec3f rvec)
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

cv::Vec3f billiards::imgproc::set_filtered_table_rot(cv::Vec3f table_rot, cv::Vec3f new_rot, float alpha, float jump_threshold)
{
    // 180도 회전한 경우, 다시 180도 돌려줍니다.
    if (norm(table_rot - new_rot) > (170.0f) * CV_PI / 180.0f) {
        new_rot = rotate_local(new_rot, {0, (float)CV_PI, 0});
    }

    if (norm(new_rot - table_rot) < jump_threshold) {
        return (1 - alpha) * table_rot + alpha * new_rot;
    }
    else {
        return new_rot;
    }
}

cv::Vec3f billiards::imgproc::set_filtered_table_pos(cv::Vec3f table_pos, cv::Vec3f new_pos, float alpha, float jump_threshold)
{
    using namespace cv;
    if (norm(new_pos - table_pos) < jump_threshold) {
        return (1 - alpha) * table_pos + alpha * new_pos;
    }
    else {
        return new_pos;
    }
}

void billiards::imgproc::project_model(img_t const& img, std::vector<cv::Point>& mapped, cv::Vec3f obj_pos, cv::Vec3f obj_rot, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, float FOV_h, float FOV_v)
{
    std::vector<cv::Vec2f> mapped_vec;
    project_model(img, mapped_vec, obj_pos, obj_rot, model_vertexes, do_cull, FOV_h, FOV_v);

    mapped.clear();
    for (auto& pt : mapped_vec) {
        mapped.emplace_back((int)pt[0], (int)pt[1]);
    }
}

void billiards::imgproc::project_contours(img_t const& img, const cv::Mat& rgb, std::vector<cv::Vec3f> model, cv::Vec3f pos, cv::Vec3f rot, cv::Scalar color, int thickness, cv::Vec2f FOV_deg)
{
    std::vector<cv::Point> mapped;
    project_model(img, mapped, pos, rot, model, true, FOV_deg[0], FOV_deg[1]);

    if (!mapped.empty()) {
        drawContours(rgb, std::vector{{mapped}}, -1, color, thickness);
    }
}

bool billiards::imgproc::get_safe_ROI_rect(cv::Mat const& mat, cv::Rect& roi)
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

float billiards::imgproc::contour_distance(std::vector<cv::Vec2f> const& ct_a, std::vector<cv::Vec2f>& ct_b)
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

        sum += min_dist;
        ct_b[min_idx] = ct_b.back();
        ct_b.pop_back();
    }

    return sqrt(sum);
}

void billiards::imgproc::plane_to_camera(img_t const& img, plane_t const& table_plane, plane_t& table_plane_camera)
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

void billiards::imgproc::get_point_coord_3d(img_t const& img, float& io_x, float& io_y, float z_metric)
{
    auto& c = img.camera;
    auto u = io_x;
    auto v = io_y;

    io_x = z_metric * ((u - c.cx) / c.fx);
    io_y = z_metric * ((v - c.cy) / c.fy);
}

std::array<float, 2> billiards::imgproc::get_uv_from_3d(img_t const& img, cv::Point3f const& coord_3d)
{
    std::array<float, 2> result;
    auto& [u, v] = result;
    auto& [x, y, z] = coord_3d;
    auto c = img.camera;

    u = (c.fx * x) / z + c.cx;
    v = (c.fy * y) / z + c.cy;

    return result;
}

float billiards::imgproc::get_pixel_length(img_t const& img, float len_metric, float Z_metric)
{
    using namespace cv;

    auto [u1, v1] = get_uv_from_3d(img, Vec3f(0, 0, Z_metric));
    auto [u2, v2] = get_uv_from_3d(img, Vec3f(len_metric, 0, Z_metric));

    return u2 - u1;
}

int billiards::imgproc::get_pixel_length_on_contact(img_t const& imdesc, plane_t plane, cv::Point pt, float length)
{
    using namespace cv;

    Vec3f far(pt.x, pt.y, 1);
    get_point_coord_3d(imdesc, far[0], far[1], 1);
    if (auto distance = plane.calc_u({}, far); distance && *distance > 0) {
        auto pxl_len = get_pixel_length(imdesc, length, *distance);
        return pxl_len;
    }

    return -1;
}

void billiards::imgproc::carve_outermost_pixels(cv::InputOutputArray io, cv::Scalar as)
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

void billiards::imgproc::project_model_points(img_t const& img, std::vector<cv::Vec2f>& mapped_contour, std::vector<cv::Vec3f>& model_vertexes, bool do_cull, std::vector<plane_t> const& planes)
{
    // 오브젝트 포인트에 frustum culling 수행
    if (do_cull) {
        // 평면 밖의 점을 모두 discard
        for (auto& plane : planes) {
            for (int i = 0; i < model_vertexes.size();) {
                if (plane.calc(model_vertexes[i]) <= 0) {
                    model_vertexes[i] = model_vertexes.back();
                    model_vertexes.pop_back();
                    continue;
                }

                ++i;
            }
        }
    }

    if (!model_vertexes.empty()) {
        // obj_pts 점을 카메라에 대한 상대 좌표로 치환합니다.
        auto [mat_cam, mat_disto] = get_camera_matx(img);

        // 각 점을 매핑합니다.
        // projectPoints(model_vertexes, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), mat_cam, mat_disto, mapped_contour);
        project_points(model_vertexes, mat_cam, mat_disto, mapped_contour);
    }
}

void billiards::imgproc::draw_circle(img_t const& img, cv::Mat& dest, float base_size, cv::Vec3f tvec_world, cv::Scalar color, int thickness)
{
    using namespace cv;
    std::vector<Vec3f> pos{{0, 0, 0}};
    std::vector<Vec2f> pt;

    project_model(img, pt, tvec_world, {0, 1, 0}, pos, false);

    float size = get_pixel_length(img, base_size, pos[0][2]);
    if (size > 1) {
        circle(dest, Point(pt[0][0], pt[0][1]), size, color, thickness);
    }
}

std::optional<billiards::imgproc::transform_estimation_result_t> billiards::imgproc::estimate_matching_transform(img_t const& img, std::vector<cv::Vec2f> const& input_param, std::vector<cv::Vec3f> model, cv::Vec3f init_pos, cv::Vec3f init_rot, transform_estimation_param_t const& p)
{
    using namespace std;
    using namespace cv;

    // 입력을 verify합니다. frustum culliing을 활용하기 때문에, 반드시 컨투어 픽셀 두 개 이상이 이미지의 경계선에 걸쳐 있어야 합니다.
    // 적어도 2개 이상의 정점이 이미지 경계 밖에 있어야 하고, 2개 이상의 정점이 경계 안에 있어야 합니다.
    auto input = input_param;
    {
        int num_in = 0, num_out = 0;
        fit_contour_to_screen(input, p.contour_cull_rect);

        // 컬링 된 컨투어 그리기
        vector<cv::Vec2i> pts;
        if (input.size() && p.render_debug_glyphs) {
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
            struct alignas(128) thread_context {
                mt19937 rand;
                vector<cv::Vec2f> ch_mapped; // 메모리 재할당 방지
                vector<cv::Vec3f> ch_model;  // 메모리 재할당 방지

                thread_context()
                    : rand((unsigned)hash<thread::id>{}(this_thread::get_id())) { }
            };
            vector<thread_context> contexts{thread::hardware_concurrency()};

            kangsw::for_each_partition(
              execution::par_unseq,
              cands.begin(),
              cands.end(),
              [&](candidate_t& elem, size_t partition_index) {
                  candidate_t cand;
                  auto& context = contexts[partition_index];
                  auto& ch_mapped = context.ch_mapped;
                  auto& ch_model = context.ch_model;
                  auto& rand = context.rand;

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

                  // 컨투어 개수가 달라도 기각함에 유의!
                  if (ch_mapped.empty() || ch_mapped.size() != input.size()) {
                      cand.error = numeric_limits<float>::max();
                  }
                  else {
                      fit_contour_to_screen(ch_mapped, p.contour_cull_rect);
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

billiards::imgproc::plane_t billiards::imgproc::plane_t::from_NP(cv::Vec3f N, cv::Vec3f P)
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

billiards::imgproc::plane_t billiards::imgproc::plane_t::from_rp(cv::Vec3f rvec, cv::Vec3f tvec, cv::Vec3f up)
{
    using namespace cv;
    auto P = tvec;
    Matx33f rotator = rodrigues(rvec);
    // Matx33f rotator;
    // Rodrigues(rvec, rotator);
    auto N = rotator * up;
    return plane_t::from_NP(N, P);
}

billiards::imgproc::plane_t& billiards::imgproc::plane_t::transform(cv::Vec3f tvec, cv::Vec3f rvec)
{
    using namespace cv;

    auto P = -N * d;
    Matx33f rotator;
    Rodrigues(rvec, rotator);

    N = rotator * N;
    P = rotator * P + tvec;

    return *this = from_NP(N, P);
}

float billiards::imgproc::plane_t::calc(cv::Vec3f const& pt) const
{
    auto v = N.mul(pt);
    auto res = v[0] + v[1] + v[2] + d;
    return res; //abs(res) < 1e-6f ? 0 : res;
}

bool billiards::imgproc::plane_t::has_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    return !!find_contact(P1, P2);
}

std::optional<float> billiards::imgproc::plane_t::calc_u(cv::Vec3f const& P1, cv::Vec3f const& P2) const
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

std::optional<cv::Vec3f> billiards::imgproc::plane_t::find_contact(cv::Vec3f const& P1, cv::Vec3f const& P2) const
{
    if (auto uo = calc_u(P1, P2); uo /*&& calc(P1) * calc(P2) < 0*/) {
        auto u = *uo;

        if (u <= 1.f && u >= 0.f) {
            return P1 + (P2 - P1) * u;
        }
    }
    return {};
}

void billiards::imgproc::filter_hsv(cv::InputArray input, cv::OutputArray output, cv::Vec3f min_hsv, cv::Vec3f max_hsv)
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

bool billiards::imgproc::is_border_pixel(cv::Rect img_size, cv::Vec2i pixel, int margin)
{
    pixel = pixel - (cv::Vec2i)img_size.tl();
    bool w = pixel[0] < margin || pixel[0] >= img_size.width - margin;
    bool h = pixel[1] < margin || pixel[1] >= img_size.height - margin;
    return w || h;
}

void billiards::imgproc::get_table_model(std::vector<cv::Vec3f>& vertexes, cv::Vec2f model_size)
{
    vertexes.clear();
    auto [half_x, half_z] = (model_size * 0.5f).val;
    vertexes.assign(
      {
        {-half_x, 0, half_z},
        {-half_x, 0, -half_z},
        {half_x, 0, -half_z},
        {half_x, 0, half_z},
      });
}

std::pair<cv::Matx33d, cv::Matx41d> billiards::imgproc::get_camera_matx(billiards::recognizer_t::frame_desc const& img)
{
    auto& p = img.camera;
    double disto[] = {0, 0, 0, 0}; // Since we use rectified image ...

    double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
    return {cv::Matx33d(M), cv::Matx41d(disto)};
}
