#include "table_search.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/ximgproc/seeds.hpp>
#include <opencv2/ximgproc/slic.hpp>

#include "kangsw/hash_index.hxx"

#pragma warning(disable : 4305, disable : 4244, disable : 4267)

struct SEEDS_setting {
    cv::Size sz;
    int num_segs;
    int num_levels;
};

using cv::ximgproc::SuperpixelSEEDS;

struct billiards::pipes::superpixel::implmentation {
    SEEDS_setting setting_cache = {};
    cv::Ptr<SuperpixelSEEDS> engine;

    cv::Mat out_array;
};

pipepp::pipe_error billiards::pipes::superpixel::invoke(pipepp::execution_context& ec, input_type const& i, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    auto color_mat = i.rgb;

    if (auto width = target_image_width(ec); width < i.rgb.cols) {
        PIPEPP_ELAPSE_SCOPE("Resize Image");

        auto size = color_mat.size();
        auto ratio = (double)width / size.width;

        size = cv::Size2d(size) * ratio;

        cv::Mat temp;
        cv::resize(color_mat, temp, size, 0, 0, cv::INTER_NEAREST);
        color_mat = temp;
    }

    bool const option_is_dirty = ec.consume_option_dirty_flag();

    if (option_is_dirty) {
        auto cls = kangsw::hash_index(color_space(ec));
        imgproc::color_space_to_flag(cls, convert_to, convert_from);
    }

    o.color_convert_to = convert_to;
    o.color_convert_from = convert_from;

    if (convert_to != -1) {
        cv::cvtColor(color_mat, color_mat, convert_to);
    }

    if (!true_SLIC_false_SEEDS(ec)) {
        auto& m = *impl_;

        auto size = color_mat.size();
        SEEDS_setting setting = {
          .sz = size,
          .num_segs = num_segments(ec),
          .num_levels = num_levels(ec),
        };

        if (option_is_dirty) {
            PIPEPP_ELAPSE_SCOPE("Recreate superpixel engine");
            m.engine = cv::ximgproc::createSuperpixelSEEDS(
              size.width, size.height, 3, setting.num_segs, setting.num_levels);

            m.setting_cache = setting;
        }

        PIPEPP_ELAPSE_BLOCK("Apply algorithm")
        {
            m.engine->iterate(color_mat, std::min(size.area() / 10, SEEDS::num_iter(ec)));
        }

        if (show_segmentation_result(ec)) {
            PIPEPP_ELAPSE_SCOPE("Visualize segmentation result");
            cv::Mat display;
            cv::Mat show;
            if (convert_from != -1) {
                cv::cvtColor(color_mat, show, convert_from);
            } else {
                show = color_mat.clone();
            }
            m.engine->getLabelContourMask(display);
            show.setTo(segmentation_devider_color(ec), display);

            PIPEPP_STORE_DEBUG_DATA("Segmentation Result", show);
        }

        m.engine->getLabels(o.labels);
    } else {
        int algos[] = {cv::ximgproc::SLICO, cv::ximgproc::MSLIC, cv::ximgproc::SLIC};
        auto algo = algos[std::clamp(algo_index_SLICO_MSLIC_SLIC(ec), 0, 2)];
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> engine;

        PIPEPP_ELAPSE_BLOCK("Create SLIC instance")
        engine = cv::ximgproc::createSuperpixelSLIC(color_mat, algo, region_size(ec), ruler(ec));

        PIPEPP_ELAPSE_BLOCK("Iterate SLIC algorithm")
        engine->iterate(SLIC::num_iter(ec));

        if (show_segmentation_result(ec)) {
            PIPEPP_ELAPSE_SCOPE("Visualize segmentation result");
            cv::Mat display;
            cv::Mat show;
            if (convert_from != -1) {
                cv::cvtColor(color_mat, show, convert_from);
            } else {
                show = color_mat.clone();
            }
            engine->getLabelContourMask(display);
            show.setTo(segmentation_devider_color(ec), display);

            PIPEPP_STORE_DEBUG_DATA("Segmentation Result", show);
        }

        engine->getLabels(o.labels);
    }

    {
        int num_labels = 0;
        for (auto v : o.labels) { num_labels = std::max(v, num_labels); }
        num_labels += 1;

        o.num_labels = num_labels;
        PIPEPP_CAPTURE_DEBUG_DATA(num_labels);
    }

    o.resized_cluster_color_mat = color_mat;
    return pipepp::pipe_error::ok;
}

billiards::pipes::superpixel::superpixel()
    : impl_(std::make_unique<implmentation>())
{
}

billiards::pipes::superpixel::~superpixel() = default;

void billiards::pipes::superpixel::link_from_previous(shared_data& sd, pipepp::execution_context& ec, input_type& o)
{
    using opt = link::preprocess;
    if (opt::enable_hsv_adjust(sd)) {
        PIPEPP_REGISTER_CONTEXT(ec);
        using namespace cv;

        Mat1b ch[3];
        Mat3b hsv;

        split(sd.hsv, ch);
        ch[2] = ch[2] * opt::v_mult(sd) + opt::v_add(sd);

        merge(ch, 3, hsv);
        cvtColor(hsv, o.rgb, cv::COLOR_HSV2RGB);
        PIPEPP_STORE_DEBUG_DATA("Adjusted RGB", o.rgb);
    } else {
        o.rgb = sd.rgb;
    }
}

void billiards::pipes::superpixel::output_handler(pipepp::pipe_error e, shared_data& sd, output_type const& o)
{
    sd.cluster_color_mat = o.resized_cluster_color_mat;
}

static void LABXY_to_RGB(cv::Mat_<cv::Vec<float, 5>> centers, cv::Mat3b& center_colors, int convert_from)
{
    cv::Mat tmp[5];
    cv::Mat3f t1;
    split(centers, tmp);
    merge(tmp, 3, t1);
    t1.convertTo(center_colors, CV_8UC3, 255);
    if (convert_from != -1) { cvtColor(center_colors, center_colors, convert_from); }
}

pipepp::pipe_error billiards::pipes::clustering::invoke(pipepp::execution_context& ec, input_type const& in, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace cv;
    using namespace std;
    using namespace imgproc;

    using Mat5f = Mat_<cv::Vec<float, 5>>;
    auto& ic = in.clusters;
    auto& cluster_color_mat = ic.resized_cluster_color_mat;
    auto& spxl_labels = out.label_2d_spxl = ic.labels;

    int num_labels = ic.num_labels;
    auto spatial_weight = sqrt(spxl_labels.size().area());

    // k-means clustering을 수행합니다.
    // 항상 우하단 픽셀이 최대값이라 가정
    spxl_sum.clear();
    spxl_sum.resize(num_labels);
    spxl_kmeans_param.clear();
    spxl_kmeans_param.resize(num_labels);
    PIPEPP_STORE_DEBUG_DATA("Initial number of superpixels", spxl_sum.size());

    auto _weights = weights_L_A_B_XY(ec);
    auto weights = concat_vec(_weights, _weights[3]);
    subvec<3, 2>(weights) /= spatial_weight;

    PIPEPP_ELAPSE_BLOCK("Calculate sum/average of all channels")
    {
        auto size = spxl_labels.size();
        auto constexpr _0_to_3 = kangsw::iota{3};
        for (auto row : kangsw::iota{size.height}) {
            for (auto col : kangsw::iota{size.width}) {
                auto color = cluster_color_mat(row, col);
                auto index = spxl_labels(row, col);

                auto& at = spxl_sum.at(index);
                auto& [l, a, b, x, y, cnt] = at.val;
                subvec<0, 3>(at) += color;
                x += col, y += row;
                ++cnt;
            }
        }

        for (auto [dst, src] : kangsw::zip(spxl_kmeans_param, spxl_sum)) {
            dst = Vec<int, 5>(src.val);
            dst /= std::max(1, src[5]);
            for (auto i : _0_to_3) { dst(i) /= 255.f; } // to values
            dst = dst.mul(weights);
        }
    }

    if (debug::show_superpixel_result(ec)) {
        PIPEPP_ELAPSE_SCOPE("Superpixel result display")
        Mat5f spxl_centers{spxl_kmeans_param, true};
        for (auto& p : spxl_centers) { p = Vec<float, 5>{p.div(weights).val}; }

        Mat3b spxl_center_colors;
        LABXY_to_RGB(spxl_centers, spxl_center_colors, ic.color_convert_from);
        auto image = index_by(spxl_center_colors, spxl_labels);
        PIPEPP_STORE_DEBUG_DATA("Superpixel result", (Mat)image);
    }

    {
        // superpixel의 label matrix의 gradient를 계산해
    }

    if (
      int n_cluster = kmeans::N_cluster(ec);
      kmeans::enabled(ec) && n_cluster < num_labels) {
        PIPEPP_ELAPSE_SCOPE("Apply k-means")
        Mat1i kmeans_labels;

        TermCriteria criteria{
          kmeans::criteria::true_EPSILON_false_ITER(ec) ? TermCriteria::EPS : TermCriteria::MAX_ITER,
          kmeans::criteria::N_iter(ec),
          kmeans::criteria::epsilon(ec),
        };
        int attempts = kmeans::attempts(ec);
        int flag = kmeans::flag_centoring_true_RANDOM_false_PP(ec) ? KMEANS_RANDOM_CENTERS : KMEANS_PP_CENTERS;

        Mat5f centers;

        cv::kmeans(spxl_kmeans_param, n_cluster, kmeans_labels, criteria, attempts, flag, centers);
        out.label_cluster_1darray = kmeans_labels;

        PIPEPP_CAPTURE_DEBUG_DATA(mat_info_str(centers));

        if (debug::show_kmeans_result(ec)) {
            for (auto& p : centers) { p = Vec<float, 5>{p.div(weights).val}; }
            PIPEPP_ELAPSE_SCOPE("k-means visualize");
            Mat3b center_colors;
            LABXY_to_RGB(centers, center_colors, ic.color_convert_from);
            auto spxl_colors = index_by(center_colors, kmeans_labels);
            auto colors = index_by(spxl_colors, spxl_labels);
            PIPEPP_STORE_DEBUG_DATA("k-means result", (Mat)colors);
        }
    }

    return pipepp::pipe_error::ok;
}

void billiards::pipes::clustering::link_from_previous(shared_data const& sd, superpixel::output_type const& o, input_type& i)
{
    i.clusters = o;
}

void billiards::pipes::clustering::output_handler(shared_data& sd, output_type const& o)
{
    sd.cluster = o;
}

pipepp::pipe_error billiards::pipes::table_contour_detection::invoke(pipepp::execution_context& ec, input_type const& in, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    using namespace std;
    using namespace cv;
    using namespace imgproc;
    namespace ks = kangsw;

    // edge 계산
    auto& labels = in.labels;
    auto& debug = *in.dbg_mat;
    Mat1b edges;
    PIPEPP_ELAPSE_BLOCK("Edge Calculation")
    {
        Mat1i expanded;
        auto size = labels.size();
        copyMakeBorder(labels, expanded, 0, 1, 0, 1, BORDER_REPLICATE);

        auto find_border = [](cv::Size size, auto const& src /* expanded */, auto&& dst) {
            for (auto& [i, j] : ks::counter(size.height, size.width)) {
                auto A = src(i, j);
                bool const is_edge
                  = A != src(i + 1, j)
                    || A != src(i, j + 1)
                    || A != src(i + 1, j + 1);

                dst(i, j) = is_edge * 255;
            }
        };

        // TODO: 한 번의 경계 계산만 수행하면, Superpixel 결과 작은 클러스터들로 인해 노이즈가 생기게 되므로 Dilate-Erode 연산 후 다시 경계선을 검출해 작은 크기의 경계들을 병합합니다.
        edges.create(expanded.size());
        edges.row(edges.rows - 1).setTo(0);
        edges.col(edges.cols - 1).setTo(0);

        find_border(size, expanded, edges);

        if (auto num_denoise = edge::pp_dilate_erode_count(ec); num_denoise) {
            PIPEPP_ELAPSE_SCOPE("Edge denoising process");
            PIPEPP_STORE_DEBUG_DATA_COND("Raw edge info before apply denoiser", expanded.clone(), debug::show_raw_edges(ec));
            UMat u0, u1;
            dilate(edges.getUMat(ACCESS_READ), u0, {}, {-1, -1}, num_denoise);
            erode(u0, u1, {}, {-1, -1}, num_denoise);
            u1.copyTo(edges);
            find_border(size, edges, edges);
        }

        edges = edges(Rect({}, size - Size(1, 1))); // 삽입한 경계선 제거
        PIPEPP_STORE_DEBUG_DATA("Calculated label edge", (Mat)edges);
    }

    // Hough 적용
    PIPEPP_ELAPSE_BLOCK("Apply Hough")
    {
        vector<cv::Vec4i> lines;

        if (hough::use_P_version(ec)) {
            HoughLinesP(edges,
                        lines,
                        hough::rho(ec),
                        hough::theta(ec) * CV_PI / 180,
                        hough::threshold(ec),
                        hough::p::min_line_len(ec),
                        hough::p::max_line_gap(ec));
        } else {
            vector<cv::Vec2f> result;
            HoughLines(edges,
                       result,
                       hough::rho(ec),
                       hough::theta(ec) * CV_PI / 180,
                       hough::threshold(ec),
                       hough::np::srn(ec),
                       hough::np::stn(ec));

            for (size_t i = 0; i < result.size(); i++) {
                float rho = result[i][0], theta = result[i][1];
                Vec4i& line = lines.emplace_back();
                auto& pt1 = subvec<0, 2>(line);
                auto& pt2 = subvec<2, 2>(line);

                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1[0] = cvRound(x0 + 1000 * (-b));
                pt1[1] = cvRound(y0 + 1000 * (a));
                pt2[0] = cvRound(x0 - 1000 * (-b));
                pt2[1] = cvRound(y0 - 1000 * (a));
            }
        }

        PIPEPP_CAPTURE_DEBUG_DATA(lines.size());

        if (debug::show_found_lines(ec)) {
            PIPEPP_ELAPSE_SCOPE("Hough visualization");
            cv::Mat vis;
            debug.size() != labels.size() ? resize(debug, vis, labels.size()) : debug.copyTo(vis);

            for (auto& line : lines) {
                auto &p0 = subvec<0, 2>(line), p1 = subvec<2, 2>(line);
                cv::line(vis, p0, p1, {0, 0, 255}, 1, LINE_AA);
            }
            PIPEPP_STORE_DEBUG_DATA("Hough line result", vis);
        }
    }

    return pipepp::pipe_error::ok;
}

bool billiards::pipes::table_contour_detection::linker(shared_data const& sd, input_type& i)
{
    if (sd.cluster.label_2d_spxl.empty() || sd.cluster.label_cluster_1darray.empty()) {
        return false;
    }

    i.labels = imgproc::index_by(sd.cluster.label_cluster_1darray, sd.cluster.label_2d_spxl);
    i.dbg_mat = &sd.debug_mat;

    return true;
}
