#include "table_search.hpp"

#include <opencv2/ximgproc/seeds.hpp>
#include <opencv2/ximgproc/slic.hpp>

//#include <memory>
//#include "table_search.hpp"
//#include "gSLICr.h"
//#include "objects/gSLICr_settings.h"
//
//using cv::Mat;
//using cv::Vec3b;
//using gSLICr::UChar4Image;
//using gSLICr::engines::core_engine;
//using gSLICr::objects::settings;
//using std::optional;
//
//struct billiards::pipes::clustering::implmentation {
//    optional<core_engine> engine;
//    settings settings_cache;
//    optional<UChar4Image> input_buffer;
//};
//
//void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
//{
//    gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);
//
//    for (int y = 0; y < outimg->noDims.y; y++)
//        for (int x = 0; x < outimg->noDims.x; x++) {
//            int idx = x + y * outimg->noDims.x;
//            outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
//            outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
//            outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
//        }
//}
//
//void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg)
//{
//    const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);
//
//    for (int y = 0; y < inimg->noDims.y; y++)
//        for (int x = 0; x < inimg->noDims.x; x++) {
//            int idx = x + y * inimg->noDims.x;
//            outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
//            outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
//            outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
//        }
//}
//
//pipepp::pipe_error billiards::pipes::clustering::invoke(pipepp::execution_context& ec, input_type const& i, output_type& o)
//{
//    PIPEPP_REGISTER_CONTEXT(ec);
//    auto& m = *impl_;
//
//    auto size = i.rgb.size();
//    settings setting = {
//      .img_size{size.width, size.height},
//      .no_segs = num_segments(ec),
//      .spixel_size = spxl_size(ec),
//      .no_iters = num_iter(ec),
//      .coh_weight = coh_weight(ec),
//      .do_enforce_connectivity = do_enforce_connectivity(ec),
//      .color_space = gSLICr::RGB,
//      .seg_method = true_segments_else_size(ec) ? gSLICr::GIVEN_NUM : gSLICr::GIVEN_SIZE,
//    };
//
//    if (memcmp(&m.settings_cache, &setting, sizeof setting) != 0) {
//        PIPEPP_ELAPSE_SCOPE("Refresh settings");
//        m.settings_cache = setting;
//        m.engine.reset();
//        m.engine.emplace(setting);
//        m.input_buffer.reset();
//        m.input_buffer.emplace(setting.img_size, true, true);
//    }
//
//    PIPEPP_ELAPSE_BLOCK("Process SLIC")
//    {
//        load_image(i.rgb, &*m.input_buffer);
//        m.engine->Process_Frame(&*m.input_buffer);
//    }
//
//    PIPEPP_ELAPSE_BLOCK("Copy labels")
//    {
//    }
//
//    if (show_segmentation_result(ec)) {
//        PIPEPP_ELAPSE_SCOPE("Show result");
//        m.engine->Draw_Segmentation_Result(&*m.input_buffer);
//        cv::Mat3b segmentation_result(size);
//        load_image(&*m.input_buffer, segmentation_result);
//        PIPEPP_CAPTURE_DEBUG_DATA(segmentation_result);
//    }
//
//    return pipepp::pipe_error::ok;
//}
//
//billiards::pipes::clustering::clustering()
//    : impl_(std::make_unique<implmentation>())
//{
//}
//
//billiards::pipes::clustering::~clustering() = default;

struct SEEDS_setting {
    cv::Size sz;
    int num_segs;
    int num_levels;
};

using cv::ximgproc::SuperpixelSEEDS;

struct billiards::pipes::clustering::implmentation {
    SEEDS_setting setting_cache = {};
    cv::Ptr<SuperpixelSEEDS> engine;
};

pipepp::pipe_error billiards::pipes::clustering::invoke(pipepp::execution_context& ec, input_type const& i, output_type& o)
{
    PIPEPP_REGISTER_CONTEXT(ec);

    if (!true_SLIC_false_SEEDS(ec)) {
        auto& m = *impl_;

        auto size = i.rgb.size();
        SEEDS_setting setting = {
          .sz = size,
          .num_segs = num_segments(ec),
          .num_levels = num_levels(ec),
        };

        if (ec.is_option_dirty()) {
            PIPEPP_ELAPSE_SCOPE("Recreate superpixel engine");
            m.engine = cv::ximgproc::createSuperpixelSEEDS(
              size.width, size.height, 3, setting.num_segs, setting.num_levels);

            m.setting_cache = setting;
            ec.clear_option_dirty_flag();
        }

        PIPEPP_ELAPSE_BLOCK("Apply algorithm")
        {
            m.engine->iterate(i.cielab, num_iter(ec));
        }

        if (show_segmentation_result(ec)) {
            PIPEPP_ELAPSE_SCOPE("Visualize segmentation result");
            cv::Mat display;
            cv::Mat show = i.rgb.clone();
            m.engine->getLabelContourMask(display);
            show.setTo(cv::Scalar{0, 255, 0}, display);

            PIPEPP_STORE_DEBUG_DATA("Segmentation Result", show);
        }

        m.engine->getLabels(o.labels);
        return pipepp::pipe_error::ok;
    }
    else {
        int algos[] = {cv::ximgproc::SLICO, cv::ximgproc::MSLIC, cv::ximgproc::SLIC};
        auto algo = algos[std::clamp(algo_index_SLICO_MSLIC_SLIC(ec), 0, 2)];
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> engine;

        PIPEPP_ELAPSE_BLOCK("Create SLIC instance")
        engine = cv::ximgproc::createSuperpixelSLIC(i.cielab, algo, region_size(ec), ruler(ec));

        PIPEPP_ELAPSE_BLOCK("Iterate SLIC algorithm")
        engine->iterate(num_iter(ec));

        if (show_segmentation_result(ec)) {
            PIPEPP_ELAPSE_SCOPE("Visualize segmentation result");
            cv::Mat display;
            cv::Mat show = i.rgb.clone();
            engine->getLabelContourMask(display);
            show.setTo(cv::Scalar{0, 255, 0}, display);

            PIPEPP_STORE_DEBUG_DATA("Segmentation Result", show);
        }

        engine->getLabels(o.labels);
        return pipepp::pipe_error::ok;
    }
}

billiards::pipes::clustering::clustering()
    : impl_(std::make_unique<implmentation>())
{
}

billiards::pipes::clustering::~clustering() = default;

void billiards::pipes::clustering::link_from_previous(shared_data const& sd, input_resize::output_type const& i, input_type& o)
{
    o.rgb = sd.rgb;
    o.hsv = sd.hsv;
    cv::cvtColor(sd.rgb, o.cielab, cv::COLOR_RGB2Lab);
}
