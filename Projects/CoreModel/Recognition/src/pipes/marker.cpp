#include "marker.hpp"
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <random>
#include <span>

#include "../amp-math/helper.hxx"
#include "../image_processing.hpp"
#include "fmt/core.h"
#include "helpers.hpp"

#undef max
#undef min

struct billiards::pipes::table_marker_finder::impl {
    // array<>
    size_t                 positive_index_fence = 0;
    std::vector<cv::Vec3f> kernel_model;
    std::vector<cv::Vec2f> _kernel_mem;

    struct {
        std::optional<concurrency::array<cv::Vec2f>> kernel_vertexes;
        std::optional<concurrency::array<float, 2>>  distances;
    } gpu;
};

void billiards::pipes::helpers::table_edge_extender::operator()(pipepp::execution_context& ec, cv::Mat& marker_area_mask)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    auto&      img               = *p_imdesc;
    bool const draw_debug_glyphs = !!debug_mat;
    auto&      debug             = *debug_mat;

    {
        vector<Vec2f> contour;

        // 테이블 평면 획득
        auto table_plane = plane_t::from_rp(table_rot, table_pos, {0, 1, 0});
        plane_to_camera(img, table_plane, table_plane);

        // contour 개수를 단순 증식합니다.
        for (int i = 0, num_insert = num_insert_contour_vertexes;
             i < table_contour.size() - 1;
             ++i) {
            auto p0 = table_contour[i];
            auto p1 = table_contour[i + 1];

            for (int idx = 0; idx < num_insert + 1; ++idx) {
                contour.push_back(p0 + (p1 - p0) * (1.f / num_insert * idx));
            }
        }

        auto& outer_contour = output.outer_contour;
        auto& inner_contour = output.inner_contour;

        outer_contour            = contour;
        inner_contour            = contour;
        auto   m                 = moments(contour);
        auto   center            = Vec2f(m.m10 / m.m00, m.m01 / m.m00);
        auto   mass              = sqrt(m.m00);
        double frame_width_outer = table_border_range_outer;
        double frame_width_inner = table_border_range_inner;

        // 각 컨투어의 거리 값에 따라, 차등적으로 밀고 당길 거리를 지정합니다.
        for (int index = 0; index < contour.size(); ++index) {
            Vec2i pt    = contour[index];
            auto& outer = outer_contour[index];
            auto& inner = inner_contour[index];

            // 거리 획득
            // auto depth = img.depth.at<float>((Point)pt);
            // auto drag_width_outer = min(300.f, get_pixel_length(img, frame_width_outer, depth));
            // auto drag_width_inner = min(300.f, get_pixel_length(img, frame_width_inner, depth));

            /*/
            float drag_width_outer = table_border_range_outer * 100;
            float drag_width_inner = table_border_range_inner * 100;
            /*/
            float drag_width_outer = get_pixel_length_on_contact(img, table_plane, pt, frame_width_outer);
            float drag_width_inner = get_pixel_length_on_contact(img, table_plane, pt, frame_width_inner);
            //*/

            // 평면과 해당 방향 시야가 이루는 각도 theta를 구하고, cos(theta)를 곱해 화면상의 픽셀 드래그를 구합니다.
            Vec3f pt_dir(pt[0], pt[1], 1);
            get_point_coord_3d(img, pt_dir[0], pt_dir[1], 1);
            pt_dir         = normalize(pt_dir);
            auto cos_theta = abs(pt_dir.dot(table_plane.N));
            drag_width_outer *= cos_theta;
            drag_width_inner *= cos_theta;
            drag_width_outer = isnan(drag_width_outer) ? 1 : drag_width_outer;
            drag_width_inner = isnan(drag_width_inner) ? 1 : drag_width_inner;
            drag_width_outer = clamp<float>(drag_width_outer, 1, 100);
            drag_width_inner = clamp<float>(drag_width_inner, 1, 100);

            auto direction = /*normalize*/ (outer - center);
            outer += direction * drag_width_outer / mass; // 질량으로 나누어 direction을 노멀라이즈합니다.
            if (!is_border_pixel({{}, marker_area_mask.size()}, inner)) {
                inner -= direction * drag_width_inner / mass;
            }
        }

        if (should_draw) {
            vector<Vec2i> drawer;
            drawer.assign(outer_contour.begin(), outer_contour.end());
            drawContours(marker_area_mask, vector{{drawer}}, -1, 255, -1);
            if (draw_debug_glyphs) {
                drawContours(debug, vector{{drawer}}, -1, {0, 0, 0}, 2);
            }

            drawer.assign(inner_contour.begin(), inner_contour.end());
            drawContours(marker_area_mask, vector{{drawer}}, -1, 0, -1);
            if (draw_debug_glyphs) {
                drawContours(debug, vector{{drawer}}, -1, {0, 0, 0}, 2);
            }
        }
    }
}

pipepp::pipe_error billiards::pipes::table_marker_finder::operator()(pipepp::execution_context& ec, input_type const& in, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto&      m          = *impl_;
    auto&      imdesc     = *in.p_imdesc;
    bool const show_debug = show_debug_mats(ec);

    if (in.contour.empty()) {
        out.marker_weight_map = cv::Mat{};
        return pipepp::pipe_error::warning;
    }

    // 옵션이 바뀐 경우 커널을 재생성합니다.
    bool const option_dirty = ec.consume_option_dirty_flag();
    if (option_dirty) {
        PIPEPP_ELAPSE_SCOPE("Regenerate Kernels");

        // 컨투어 랜덤 샘플 피봇 재생성
        auto                                  _positive                = kernel::positive_area(ec);
        auto                                  _negative                = kernel::negative_area(ec);
        auto                                  positive_integral_radius = kernel::generator_positive_radius(ec);
        auto                                  negative_integral_radius = kernel::generator_negative_radius(ec);
        std::mt19937                          rand(kernel::random_seed(ec));
        std::uniform_real_distribution<float> positive(_positive[0], _positive[1]);
        std::uniform_real_distribution<float> negative(_negative[0], _negative[1]);

        auto kg = helpers::kernel_generator{
          .positive                 = positive,
          .negative                 = negative,
          .positive_integral_radius = kernel::generator_positive_radius(ec),
          .negative_integral_radius = kernel::generator_negative_radius(ec),
          .random_seed              = kernel::random_seed(ec),
          .show_debug               = show_debug,
          .kernel_view_size         = debug::kernel_view_size(ec),
        };

        auto vtxs = kg(ec);

        // 반지름을 곱합니다.
        for (auto radius = marker_radius(ec); auto& vtx : vtxs) {
            vtx *= radius;
        }
        m.positive_index_fence = kg.output.positive_index_fence;
        m.kernel_model         = std::move(vtxs);
    }

    // 커널을 테이블과의 각도에 따라 회전시키고, 1미터 거리를 기준으로 투영합니다.
    // 이를 통해 원형의 마커를 회전에 따라 알맞은 각도로 적용할 수 있습니다.
    auto& rotated_kernel = m._kernel_mem;
    PIPEPP_ELAPSE_BLOCK("Project points by view")
    {
        // 1 meter 거리에 대해 모든 포인트를 투사합니다.
        // TODO : Positive kernel만 회전시키기. Negative kernel은 그대로 둠!
        using namespace imgproc;
        decltype(m.kernel_model) vertices;
        vertices.assign(m.kernel_model.begin(), m.kernel_model.begin() + m.positive_index_fence);

        cv::Vec3f local_loc = {}, local_rot = in.init_table_rot;
        world_to_camera(imdesc, local_rot, local_loc);

        // 항상 카메라에서 1미터 떨어진 위치에서의 크기를 계산합니다.
        local_loc = {0, 0, 1};

        for (auto  rotation = rodrigues(local_rot);
             auto& vt : vertices) {
            vt = rotation * vt + local_loc;
        }

        rotated_kernel.clear();
        project_model_local(imdesc, rotated_kernel, vertices, false, {});

        // 커널을 카메라 중심으로 매핑
        for (cv::Vec2f cam_center(imdesc.camera.cx, imdesc.camera.cy);
             auto&     vt : rotated_kernel) {
            vt -= cam_center;
        }

        // negative 인덱스 append
        vertices.insert(vertices.end(), m.kernel_model.begin() + m.positive_index_fence, m.kernel_model.end());
        std::span nkernel = vertices;
        nkernel           = nkernel.subspan(m.positive_index_fence);
        auto marker_rad   = marker_radius(ec);
        auto pixel_radius = imgproc::get_pixel_length(imdesc, marker_rad, 1);
        for (auto& vt : nkernel) {
            vt = vt / marker_rad * pixel_radius;
            std::swap(vt[1], vt[2]);
            rotated_kernel.push_back(subvec<0, 2>(vt));
        }

        if (show_debug) {
            PIPEPP_ELAPSE_SCOPE("Kernel visualization");
            auto       scale  = debug::kernel_view_size(ec);
            auto       mult   = debug::current_kernel_view_scale(ec);
            auto       radius = std::max(1, (int)scale / 100);
            cv::Mat3b  kernel_view(scale, scale, {0, 0, 0});
            cv::Point  center(scale / 2, scale / 2);
            cv::Scalar colors[] = {{0, 255, 0}, {0, 0, 255}};

            for (auto idx : kangsw::counter(rotated_kernel.size())) {
                auto fpt = rotated_kernel[idx];
                fpt      = fpt * mult;

                cv::circle(kernel_view, center + (cv::Point)(cv::Vec2i)fpt, radius, colors[idx >= m.positive_index_fence]);
            }

            PIPEPP_STORE_DEBUG_DATA("Current kernel", (cv::Mat)kernel_view);
        }
    }

    // 커널을 유효한 부분에서만 계산할 수 있도록, 먼저 컨투어를 유효 영역으로 확장합니다.
    cv::Mat1b area_mask(in.domain.size(), 0);
    cv::Rect  roi;
    cv::Mat3b pp_filter_domain;
    PIPEPP_ELAPSE_BLOCK("Table contour area extension")
    {
        if (in.all_none_blue.empty() == false) {
            PIPEPP_STORE_DEBUG_DATA("All none-blue area mask", (cv::Mat)in.all_none_blue);
            roi              = cv::Rect({}, in.domain.size());
            area_mask        = in.all_none_blue;
            pp_filter_domain = in.domain;
        } else {
            helpers::table_edge_extender tee = {
              .table_rot                   = in.init_table_rot,
              .table_pos                   = in.init_table_pos,
              .p_imdesc                    = in.p_imdesc,
              .table_contour               = in.contour,
              .num_insert_contour_vertexes = pp::num_inserted_contours(ec),
              .table_border_range_outer    = pp::marker_range_outer(ec),
              .table_border_range_inner    = pp::marker_range_inner(ec),
              .debug_mat                   = (cv::Mat*)&in.debug,
            };
            tee(ec, area_mask);
            PIPEPP_STORE_DEBUG_DATA_COND("Marker Area Mask", (cv::Mat)area_mask, show_debug);

            roi = cv::boundingRect(tee.output.outer_contour);
            imgproc::get_safe_ROI_rect(in.domain, roi);
            area_mask        = area_mask(roi);
            pp_filter_domain = in.domain(roi);
        }
    }

    // 유효 픽셀만을 필터링합니다. 두 가지 메소드 중 하나를 적용하게 됩니다.
    // (델타 색상 또는 범위 필터)
    std::vector<cv::Vec2i> valid_marker_pixels;
    PIPEPP_ELAPSE_BLOCK("Filter valid marker pixels")
    {
        using namespace imgproc;
        cv::Mat1b valid_pxls;

        if (in.lightness.empty()) {
            // 0번 메소드 시 lightness가 지정되지 않습니다.
            // 0번 메소드 ==> Range filter 후 모든 픽셀 선택
            range_filter(pp_filter_domain, valid_pxls, filter::method0::range_lo(ec), filter::method0::range_hi(ec));
        } else {
            // 다른 메소드에서는 lightness가 지정됩니다.
            // 먼저 lightness Mat에 2D 필터 적용
            float edge_kernel[3][3] = {{-1, -1, -1},
                                       {-1, +8, -1},
                                       {-1, -1, -1}};

            cv::Mat   lightness;
            cv::Mat1f edges(roi.size());

            // 컨볼루션 수행
            if (filter::method1::enable_gpu(ec) == false) {
                PIPEPP_ELAPSE_SCOPE("CPU Convolution");
                auto inner_counter = kangsw::counter(3, 3);
                in.lightness(roi).convertTo(lightness, CV_32FC1, 1 / 255.0);
                cv::filter2D(lightness, edges, -1, cv::Matx33f(*edge_kernel));
            } else {
                PIPEPP_ELAPSE_SCOPE("GPU Convolution");
                cv::copyMakeBorder(in.lightness(roi), lightness, 1, 1, 1, 1, cv::BORDER_DEFAULT);
                lightness.convertTo(lightness, CV_32FC1, 1 / 255.0);

                using namespace concurrency;
                array_view<float, 2> gpu_lightness(lightness.rows, lightness.cols, &lightness.at<float>(0));
                array_view<float, 2> gpu_edges(roi.height, roi.width, &edges(0));
                array_view<float, 2> gpu_kernel(3, 3, &edge_kernel[0][0]);
                gpu_edges.discard_data();

                parallel_for_each(
                  gpu_edges.extent,
                  [=](index<2> idx) restrict(amp) {
                      int row = idx[0];
                      int col = idx[1];

                      float sum = 0;
                      for (int i = -1; i < 2; ++i) {
                          for (int j = -1; j < 2; ++j) {
                              sum += gpu_lightness(row + i + 1, col + j + 1) * gpu_kernel(i + 1, j + 1);
                          }
                      }

                      gpu_edges(row, col) = sum;
                  });

                gpu_edges.synchronize();
            }

            if (show_debug) {
                PIPEPP_STORE_DEBUG_DATA("Source Image", (cv::Mat)in.lightness);
                PIPEPP_STORE_DEBUG_DATA("Edge of lightness", (cv::Mat)edges);
            }

            // constant threshold 값으로 이진화한 뒤, dilate-erode 연산을 적용해 마커 중심을 채웁니다.
            auto thres = filter::method1::threshold(ec);
            valid_pxls = edges > thres;

            if (auto n = filter::method1::holl_fill_num_dilate(ec)) {
                cv::dilate(valid_pxls, valid_pxls, {}, {-1, -1}, n);
            }
            if (auto n = filter::method1::holl_fill_num_erode(ec)) {
                cv::erode(valid_pxls, valid_pxls, {}, {-1, -1}, n);
            }

            if (show_debug) {
            }
        }

        valid_pxls = valid_pxls & area_mask;
        cv::findNonZero(valid_pxls, valid_marker_pixels);
        if (show_debug) {
            PIPEPP_STORE_DEBUG_DATA("Selected valid pixels", (cv::Mat)valid_pxls);
        }

        PIPEPP_STORE_DEBUG_DATA("Number of candidate pixels", valid_marker_pixels.size());
    }

    if (valid_marker_pixels.empty()) {
        return pipepp::pipe_error::warning;
    }

    // 각각의 유효 마커 픽셀을 iterate해, 거리를 계산합니다.
    std::vector<float> distances(valid_marker_pixels.size());
    PIPEPP_ELAPSE_BLOCK("Create distance buffer")
    {
        using namespace imgproc;
        using namespace cv;

        auto table_plane = plane_t::from_rp(in.init_table_rot, in.init_table_pos, {0, 1, 0});
        plane_to_camera(imdesc, table_plane, table_plane);

        for (auto idx : kangsw::rcounter(valid_marker_pixels.size())) {
            auto pos = valid_marker_pixels[idx];

            Point pt = (Point)pos + roi.tl();
            Vec3f vec(pt.x, pt.y, 1.f);
            get_point_coord_3d(imdesc, vec[0], vec[1], vec[2]);
            if (auto uo = table_plane.calc_u({}, vec)) {
                distances[idx] = *uo;
            } else { // invalid한 평면에 있다면 제거합니다.
                kangsw::swap_remove(valid_marker_pixels, idx);
                kangsw::swap_remove(distances, idx);
            }
        }

        if (show_debug) {
            cv::Mat1f depths(roi.size(), 0);
            auto      mult = debug::depth_view_multiply(ec);
            for (auto [pos, depth] : kangsw::zip(valid_marker_pixels, distances)) {
                depths((cv::Point)pos) = depth * mult;
            }

            PIPEPP_STORE_DEBUG_DATA("Marker area depths", (Mat)depths);
        }
    }

    // 희소 커널 콘볼루션을 수행합니다.
    // 후보 개수 M, 커널 개수 N인 M by N extent에서 계산을 수행합니다.
    // 이후, 누적 연산은 M by 1(또는 타일 개수?) extent에서 수행
    //
    // 알고리즘 개요
    //      1) intermediate[m, n] = colorDist(domain[n.(x,y) / m.distance], pvtColor)
    //      2) weight[m] = sum(intermediate[m, :]
    PIPEPP_ELAPSE_BLOCK("Apply sparse kernel convolution")
    if (valid_marker_pixels.empty() == false) {
        using namespace concurrency;
        using namespace graphics;
        using namespace kangsw;
        int       M = valid_marker_pixels.size();
        int       N = rotated_kernel.size();
        cv::Mat3f domainf;
        in.conv_domain(roi).convertTo(domainf, CV_32FC3, 1 / 255.0);
        std::vector<float> suitabilities(M);

        array<float, 2> u_interm_suits(M, N);

        // [M] domain
        array_view<int_2> u_indexes(M, ptr_cast<int_2>(&valid_marker_pixels[0]));
        array_view<float> u_dists(M, distances.data());
        array_view<float> u_suit(M, &suitabilities[0]);
        u_suit.discard_data();

        // [N] domain
        array_view<float_2> u_kernel(N, ptr_cast<float_2>(&rotated_kernel[0]));

        // color domain
        array_view<float_3, 2> u_domain(domainf.rows, domainf.cols, ptr_cast<float_3>(&domainf(0)));

        // constants
        int_2   img_size(domainf.cols, domainf.rows);
        float_3 color_pivot           = value_cast<float_3>((cv::Vec3f)filter::convolution::pivot_color(ec) / 255.f);
        float_3 color_weight          = value_cast<float_3>(filter::convolution::pivot_color_weight(ec));
        float   err_base              = filter::convolution::color_distance_error_base(ec);
        int     positive_kernel_fence = m.positive_index_fence;
        float   neg_weight            = filter::convolution::negative_kernel_weight(ec);

        // Step 1. 각 커널의 구성 픽셀에 대한 suitability를 계산합니다.
        parallel_for_each(
          u_interm_suits.get_extent(),
          [=, &u_interm_suits](index<2> idx) restrict(amp) {
              auto _m       = idx[0];
              auto _n       = idx[1];
              auto offset   = u_indexes[_m];
              auto distance = u_dists[_m];
              auto kpos     = u_kernel[_n];

              // auto index = int_2((float_2)offset + kpos / distance);
              float_2 offsetf(offset.x, offset.y);
              offsetf = offsetf + (kpos / distance);
              int_2 index(offsetf.x, offsetf.y);
              if ((0 <= index.x && index.x < img_size.x)
                  && (0 <= index.y && index.y < img_size.y)) {
                  // Weight를 계산합니다.
                  namespace fm = fast_math;
                  auto color   = u_domain(index.y, index.x);
                  auto dist_3  = (color - color_pivot);
                  dist_3       = dist_3 * dist_3 * color_weight;
                  auto dist    = (dist_3.x + dist_3.y + dist_3.z);

                  auto suitability    = fm::pow(err_base, -dist);
                  u_interm_suits(idx) = suitability * (_n >= positive_kernel_fence ? -neg_weight : 1.0f);
              } else {
                  u_interm_suits(idx) = 0.f;
              }
          });

        // 중간 결과의 각 커널 채널을 iterate해, 누산합니다.
        float f_positive_kernels = positive_kernel_fence;

        parallel_for_each(
          extent<1>(M),
          [=, u_interm = array_view{u_interm_suits}](index<1> idx) restrict(amp) {
              float sum = 0;
              auto  i   = idx[0];
              for (int n = 0; n < N; ++n) {
                  sum += u_interm(i, n);
              }
              u_suit(idx) = sum / f_positive_kernels;
          });

        u_suit.synchronize();

        cv::Mat1f suits(in.domain.size());
        for (auto [idx, value] : zip(valid_marker_pixels, suitabilities)) {
            suits(roi.tl() + (cv::Point)idx) = value;
        }

        out.marker_weight_map = suits;
        auto mult             = debug::suitability_view_multiply(ec);
        if (show_debug) { PIPEPP_STORE_DEBUG_DATA("Marker suitability view", (cv::Mat)(suits * mult)); }
    }

    return {};
}

void billiards::pipes::table_marker_finder::link(shared_data& sd, pipepp::execution_context& ec, input_type& i, pipepp::detail::option_base const& opt)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    if (sd.table.contour.empty()) {
        i.contour = {};
        return;
    }

    i.init_table_pos = sd.table.pos;
    i.init_table_rot = sd.table.rot;

    i.debug    = sd.debug_mat;
    i.p_imdesc = &sd.imdesc_bkup;

    i.contour = sd.table.contour;
    sd.get_marker_points_model(i.marker_model);

    auto colorspace = (filter::filter_color_space(opt));
    i.domain        = sd.retrieve_image_in_colorspace(colorspace);
    i.conv_domain   = sd.retrieve_image_in_colorspace(filter::convolution_color_space(opt));

    if (filter::method(opt) > 0) {
        cv::Mat split[3];
        cv::split(i.domain, split);

        using namespace kangsw::literals;
        switch (kangsw::hash_index(colorspace)) {
            case "YCrCb"_hash:
            case "Lab"_hash:
            case "Luv"_hash: [[fallthrough]];
            case "YUV"_hash: i.lightness = split[0]; break;
            case "HLS"_hash: i.lightness = split[1]; break;
            case "HSV"_hash: i.lightness = split[2]; break;
            default: i.lightness = {}; break;
        }
    } else {
        i.lightness = {};
    }

    if (pp::use_all_non_blue_area(opt)) {
        PIPEPP_ELAPSE_SCOPE("Table Marker Finder Preprocessing");
        // 존재하는 모든 non-blue 경계선을 확장합니다.
        using namespace std;
        using namespace cv;
        auto edge = sd.table_filtered_edge;

        Mat1b                 mask(i.domain.size(), 0);
        vector<vector<Vec2i>> all_contours;
        vector<Vec2i>         approxed;
        vector<Vec2f>         approxedf;

        findContours(edge, all_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<vector<Vec2i>> inners;
        vector<vector<Vec2i>> outers;
        inners.reserve(all_contours.size()), outers.reserve(all_contours.size());

        helpers::table_edge_extender ext = {
          .table_rot                   = sd.table.rot,
          .table_pos                   = sd.table.pos,
          .p_imdesc                    = &sd.imdesc_bkup,
          .table_contour               = {},
          .num_insert_contour_vertexes = pp::num_inserted_contours(opt),
          .table_border_range_outer    = pp::marker_range_outer(opt),
          .table_border_range_inner    = pp::marker_range_inner(opt),
          .should_draw                 = false};

        auto eps = pp::m2::approx_epsilon(opt);
        for (auto& contour : all_contours) {
            approxPolyDP(contour, approxed, eps, true);
            if (approxed.size() < 3) { continue; }
            approxedf.assign(approxed.begin(), approxed.end());
            ext.table_contour = approxedf;
            ext(ec, mask);

            inners.emplace_back().assign(ext.output.inner_contour.begin(), ext.output.inner_contour.end());
            outers.emplace_back().assign(ext.output.outer_contour.begin(), ext.output.outer_contour.end());

            ext.output.inner_contour.clear();
            ext.output.outer_contour.clear();
        }

        drawContours(mask, outers, -1, 255, -1);
        drawContours(mask, inners, -1, 0, -1);

        i.all_none_blue = mask;
    } else {
        i.all_none_blue = {};
    }
}

billiards::pipes::table_marker_finder::table_marker_finder()
    : impl_(std::make_unique<impl>())
{
}

billiards::pipes::table_marker_finder::~table_marker_finder() = default;

// ----------------------------------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------------------------------

using namespace concurrency::graphics::direct3d;
template <typename Ty_, int Rank_ = 1> using gpu_array = concurrency::array<Ty_, Rank_>;
using std::optional;

namespace billiards {
namespace {

template <typename Ty_>
Ty_ get_align(Ty_ target, int64_t align)
{
    return ((target + align - 1) / align) * align;
}

constexpr auto randgen(uint32_t init_seed) __GPU
{
    return [seed = init_seed]() mutable __GPU {
        seed     = mathf::wang_hash(seed);
        auto ret = seed / float(UINT32_MAX);
        return ret * (seed & 1 ? 1.f : -1.f);
    };
}

constexpr auto rand_sign(uint32_t init_seed) __GPU
{

    return [seed = init_seed]() mutable __GPU {
        seed = mathf::wang_hash(seed);
        return 1.0f - 2.f * float(seed & 1);
    };
}

auto transform_point(float3& io_pt, float3 tvec, float3 rvec) __GPU
{
    io_pt = mathf::rodrigues(rvec) * io_pt + tvec;
}

} // namespace
} // namespace billiards

using concurrency::array_view;

struct billiards::pipes::marker_solver_gpu::impl_type {
    optional<gpu_array<float, 2>> _mbuf_interm_suit; // tvec by rvec 크기의 버퍼입니다.
    optional<gpu_array<int2, 2>>  _mbuf_interm_best; // tvec by rvec 크기의 버퍼입니다.
    optional<gpu_array<float3>>   _mbuf_tvecs;
    optional<gpu_array<float3>>   _mbuf_rvecs;
    optional<gpu_array<float3>>   _mbuf_model;

    void _generate_vectors(
      cv::Vec3f init_location, cv::Vec3f init_rotation,
      float var_loc, float var_rot, float var_axis)
    {
        using namespace concurrency;
        using namespace kangsw::misc;

        float  irotlen   = cv::norm(init_rotation);
        uint   seed      = seed_;
        float3 iloc      = value_cast<float3>(init_location);
        float3 irotn     = value_cast<float3>(init_rotation) / irotlen;
        auto   dst_tvecs = array_view(*_mbuf_tvecs);
        auto   dst_rvecs = array_view(*_mbuf_rvecs);

        // 테이블의 노멀을 기준으로 회전시킵니다.
        float3 irotup = value_cast<float3>(imgproc::rodrigues(init_rotation) * cv::Vec3f{0, 1, 0});

        parallel_for_each(
          dst_tvecs.extent,
          [=] //
          (index<1> idx) __GPU {
              auto r = randgen(seed + idx[0]);

              if (idx[0] == 0) {
                  dst_tvecs[idx] = {};
                  return;
              }

              float3 tvec;
              tvec.x         = r() * var_loc;
              tvec.y         = r() * var_loc;
              tvec.z         = r() * var_loc;
              dst_tvecs[idx] = tvec + iloc;
          });

        parallel_for_each(
          dst_rvecs.extent,
          [=] //
          (index<1> idx) __GPU {
              auto r = randgen(seed + idx[0] * seed);

              if (idx[0] == 0) {
                  dst_rvecs[idx] = {};
                  return;
              }

              float3 rvaxis;
              rvaxis.x = r() * var_axis;
              rvaxis.y = r() * var_axis;
              rvaxis.z = r() * var_axis;

              float3 rvec    = mathf::normalize(irotup + rvaxis);
              dst_rvecs[idx] = irotn * irotlen + rvec * r() * var_rot;
          });

        seed_ = mathf::wang_hash(seed_);
    }

    float _perform_solve(cv::Matx33f           proj_mat,
                         array_view<float, 2>& weight_map,
                         cv::Vec3f&            tvec_best,
                         cv::Vec3f&            rvec_best)
    {
        using namespace concurrency;
        using namespace kangsw::misc;
        using namespace mathf::types;

        array_view u_interm_suit = *_mbuf_interm_suit;
        array_view u_interm_best = *_mbuf_interm_best;
        array_view u_model       = *_mbuf_model;
        array_view u_tvecs       = *_mbuf_tvecs;
        array_view u_rvecs       = *_mbuf_rvecs;

        float3             tv_rv_suit_00[3];
        array_view<float3> u_best(3, tv_rv_suit_00);
        u_best.discard_data();

        int const n_models = u_model.extent[0];
        auto      img_size = weight_map.extent;

        matx33f proj = value_cast<matx33f>(proj_mat);

        parallel_for_each(
          extent<2>(u_tvecs.extent[0], u_rvecs.extent[0]).tile<TILE_SIZE, TILE_SIZE>(),
          [=](tiled_index<TILE_SIZE, TILE_SIZE> tidx) __GPU_ONLY //
          {
              tile_static float local_suits[TILE_SIZE][TILE_SIZE];
              {
                  auto   gidx     = tidx.global;
                  float3 tvec     = u_tvecs[gidx[0]];
                  float3 rvec     = u_rvecs[gidx[1]];
                  float  suit_sum = 0.f;

                  for (int midx = 0; midx < n_models; ++midx) {
                      auto mpos = u_model[midx];
                      transform_point(mpos, tvec, rvec);
                      int2 sample((mpos = proj * mpos).xy / mpos.z);

                      if (0 <= sample.x && sample.x < img_size[1] && //
                          0 <= sample.y && sample.y < img_size[0])   //
                      {
                          suit_sum += weight_map(sample.y, sample.x);
                      }
                  }

                  local_suits[tidx.local[0]][tidx.local[1]] = suit_sum;
              }

              tidx.barrier.wait_with_all_memory_fence();

              // Find local best
              if (tidx.local[0] == 0 && tidx.local[1] == 0) {
                  int   best_t = 0, best_r = 0;
                  float max_suit = 0;

                  for (int t = 0; t < TILE_SIZE; ++t) {
                      for (int r = 0; r < TILE_SIZE; ++r) {
                          if (auto suit = local_suits[t][r];
                              suit > max_suit) //
                          {
                              max_suit = suit;
                              best_t   = t;
                              best_r   = r;
                          }
                      }
                  }

                  int t0 = tidx.tile[0];
                  int t1 = tidx.tile[1];
                  best_t = best_t + TILE_SIZE * t0;
                  best_r = best_r + TILE_SIZE * t1;

                  u_interm_suit(t0, t1) = max_suit;
                  auto& r               = u_interm_best(t0, t1);
                  r.x                   = best_r;
                  r.y                   = best_t;
              }

              tidx.barrier.wait_with_all_memory_fence();
              if (tidx.global[0] == 0 && tidx.global[1] == 0) {
                  auto  bests  = u_interm_suit.extent;
                  int   best_0 = 0, best_1 = 0;
                  float max_suit = 0;

                  for (int tile_i = 0; tile_i < bests[0]; ++tile_i) {
                      for (int tile_j = 0; tile_j < bests[1]; ++tile_j) {
                          if (auto suit = u_interm_suit(tile_i, tile_j);
                              suit > max_suit) //
                          {
                              best_0 = tile_i, best_1 = tile_j;
                              max_suit = suit;
                          }
                      }
                  }

                  auto best_idx = u_interm_best(best_0, best_1);
                  u_best[0]     = u_tvecs[best_idx.y];
                  u_best[1]     = u_rvecs[best_idx.x];
                  u_best[2].x   = max_suit;
                  u_best[2].yz  = {};
              }
              tidx.barrier.wait_with_all_memory_fence();
          });

        u_best.synchronize();

        tvec_best = value_cast<cv::Vec3f>(tv_rv_suit_00[0]);
        rvec_best = value_cast<cv::Vec3f>(tv_rv_suit_00[1]);
        return tv_rv_suit_00[2].x;
    }

private:
    int seed_ = 0;
};

void billiards::pipes::marker_solver_gpu::operator()(pipepp::execution_context& ec, input_type const& i, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);

    // 출력 초기화
    o.confidence = 0;

    if (ec.consume_option_dirty_flag()) {
        // 변경된 옵션에 따라 버퍼 재생성
        int tvec_vars = get_align(std::max(1u, solve::num_location_cands(ec)), TILE_SIZE);
        int rvec_vars = get_align(std::max(1u, solve::num_rotation_cands(ec)), TILE_SIZE);

        _m->_mbuf_tvecs.emplace(tvec_vars);
        _m->_mbuf_rvecs.emplace(rvec_vars);

        _m->_mbuf_interm_suit.emplace(tvec_vars / TILE_SIZE, rvec_vars / TILE_SIZE);
        _m->_mbuf_interm_best.emplace(tvec_vars / TILE_SIZE, rvec_vars / TILE_SIZE);

        PIPEPP_STORE_DEBUG_DATA("Number of TVEC cands", tvec_vars);
        PIPEPP_STORE_DEBUG_DATA("Number of RVEC cands", rvec_vars);
        PIPEPP_STORE_DEBUG_DATA("Number of all candidates", tvec_vars * rvec_vars);
    }

    if (i.marker_model.empty()) { return; }
    if (i.marker_weight_map.empty()) { return; }

    PIPEPP_ELAPSE_BLOCK("Perform match")
    {
        // Upload model
        {
            auto& md = i.marker_model;
            _m->_mbuf_model.emplace(int(md.size()), kangsw::ptr_cast<float3>(md.data()));
        }
        auto  tpos     = i.init_local_table_pos;
        auto  trot     = i.init_local_table_rot;
        float cur_suit = solve::suitability_threshold(ec);

        float var_loc  = solve::var_location(ec);
        float var_rot  = solve::var_rotation_deg(ec) * CV_PI / 180.0;
        float var_axis = solve::var_axis(ec);

        float const narrow_loc = solve::location_narrow_rate(ec);
        float const narrow_rot = solve::rotation_narrow_rate(ec);
        float const narrow_axe = solve::axis_narrow_rate(ec);

        auto& imdesc = *i.p_imdesc;
        auto  _map   = i.marker_weight_map;
        PIPEPP_STORE_DEBUG_DATA("Input", (cv::Mat)_map);

        if (_map.isContinuous() == false) _map = _map.clone();
        array_view<float, 2> u_weight_map(_map.rows, _map.cols, &_map(0));

        for (int iter = 0, max_iter = solve::num_iteration(ec);
             iter < max_iter; ++iter) //
        {
            PIPEPP_ELAPSE_SCOPE_DYNAMIC(fmt::format("Iteration {}", iter).c_str());

            _m->_generate_vectors(tpos, trot, var_loc, var_rot, var_axis);

            auto prev_pos = tpos, prev_rot = trot;
            auto suitability
              = _m->_perform_solve(imgproc::get_camera_matx(imdesc).first, u_weight_map, tpos, trot)
                / i.marker_model.size();

            PIPEPP_STORE_DEBUG_DATA_DYNAMIC(fmt::format("Accuracy ({})", iter).c_str(), suitability);

            {
                using namespace cv;
                using namespace imgproc;
                auto  table_pos = tpos;
                auto  table_rot = trot;
                auto& vertexes  = i.marker_model;

                camera_to_world(*i.p_imdesc, table_rot, table_pos);
                auto world_tr = get_transform_matx_fast(table_pos, table_rot);
                for (auto pt : vertexes) {
                    Vec4f pt4;
                    (Vec3f&)pt4 = pt;
                    pt4[3]      = 1.0f;

                    pt4 = world_tr * pt4;
                    draw_circle(*i.p_imdesc, (Mat&)i.debug_mat, 0.01f, (Vec3f&)pt4, {255, 0, 0}, 2);
                }

                PIPEPP_STORE_DEBUG_DATA("Result", (Mat)i.debug_mat.clone());
            }

            if (suitability < cur_suit) {
                tpos = prev_pos;
                trot = prev_rot;
                break;
            }

            cur_suit = suitability;
            var_loc *= narrow_loc;
            var_rot *= narrow_rot;
            var_axis *= narrow_axe;
        }

        o.local_table_pos = tpos;
        o.local_table_rot = trot;
        o.confidence      = cur_suit;
    }
}

void billiards::pipes::marker_solver_gpu::link(shared_data& sd, table_marker_finder::output_type const& o, input_type& i)
{
    i.p_imdesc             = &sd.imdesc_bkup;
    i.init_local_table_pos = sd.table.pos;
    i.init_local_table_rot = sd.table.rot;

    i.marker_weight_map = o.marker_weight_map;
    i.debug_mat         = sd.debug_mat;

    imgproc::world_to_camera(sd.imdesc_bkup, i.init_local_table_rot, i.init_local_table_pos);
    sd.get_marker_points_model(i.marker_model);
}

billiards::pipes::marker_solver_gpu::marker_solver_gpu()
    : _m(std::make_unique<impl_type>())
{
}

billiards::pipes::marker_solver_gpu::~marker_solver_gpu() = default;
