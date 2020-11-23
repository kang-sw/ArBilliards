#include "table_search.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/ximgproc/seeds.hpp>
#include <opencv2/ximgproc/slic.hpp>

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
    auto cielab = i.cielab;

    if (auto width = target_image_width(ec); width < i.cielab.cols) {
        PIPEPP_ELAPSE_SCOPE("Resize Image");

        auto size = cielab.size();
        auto ratio = (double)width / size.width;

        size = cv::Size2d(size) * ratio;
        cv::resize(cielab, impl_->out_array, size, 0, 0, cv::INTER_NEAREST);
        cielab = impl_->out_array;
    }

    if (!true_SLIC_false_SEEDS(ec)) {
        auto& m = *impl_;

        auto size = cielab.size();
        SEEDS_setting setting = {
          .sz = size,
          .num_segs = num_segments(ec),
          .num_levels = num_levels(ec),
        };

        if (ec.consume_option_dirty_flag()) {
            PIPEPP_ELAPSE_SCOPE("Recreate superpixel engine");
            m.engine = cv::ximgproc::createSuperpixelSEEDS(
              size.width, size.height, 3, setting.num_segs, setting.num_levels);

            m.setting_cache = setting;
        }

        PIPEPP_ELAPSE_BLOCK("Apply algorithm")
        {
            m.engine->iterate(cielab, SEEDS::num_iter(ec));
        }

        if (show_segmentation_result(ec)) {
            PIPEPP_ELAPSE_SCOPE("Visualize segmentation result");
            cv::Mat display;
            cv::Mat show;
            cv::cvtColor(cielab, show, cv::COLOR_Lab2RGB);
            m.engine->getLabelContourMask(display);
            show.setTo(segmentation_devider_color(ec), display);

            PIPEPP_STORE_DEBUG_DATA("Segmentation Result", show);
        }

        m.engine->getLabels(o.labels);
        return pipepp::pipe_error::abort;
    }
    else {
        int algos[] = {cv::ximgproc::SLICO, cv::ximgproc::MSLIC, cv::ximgproc::SLIC};
        auto algo = algos[std::clamp(algo_index_SLICO_MSLIC_SLIC(ec), 0, 2)];
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> engine;

        PIPEPP_ELAPSE_BLOCK("Create SLIC instance")
        engine = cv::ximgproc::createSuperpixelSLIC(cielab, algo, region_size(ec), ruler(ec));

        PIPEPP_ELAPSE_BLOCK("Iterate SLIC algorithm")
        engine->iterate(SLIC::num_iter(ec));

        if (show_segmentation_result(ec)) {
            PIPEPP_ELAPSE_SCOPE("Visualize segmentation result");
            cv::Mat display;
            cv::Mat show;
            cv::cvtColor(cielab, show, cv::COLOR_Lab2RGB);
            engine->getLabelContourMask(display);
            show.setTo(segmentation_devider_color(ec), display);

            PIPEPP_STORE_DEBUG_DATA("Segmentation Result", show);
        }

        engine->getLabels(o.labels);
        return pipepp::pipe_error::ok;
    }
}

billiards::pipes::superpixel::superpixel()
    : impl_(std::make_unique<implmentation>())
{
}

billiards::pipes::superpixel::~superpixel() = default;

void billiards::pipes::superpixel::link_from_previous(shared_data const& sd, input_resize::output_type const& i, input_type& o)
{
    cv::cvtColor(sd.rgb, o.cielab, cv::COLOR_RGB2Lab);
}
