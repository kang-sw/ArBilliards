#include "recognition.hpp"
#include <memory>
#include <exception>
#include "pipes/recognizer.hpp"
#include "pipepp/pipeline.hpp"

class billiards::recognizer_t::implementation
{
public:
    recognizer_t& self;
    std::shared_ptr<pipepp::pipeline<pipes::shared_data, pipes::input_resize>> pipeline;
    std::optional<frame_desc> param_pending;
    kangsw::spinlock param_pending_lock;

    std::shared_ptr<pipes::shared_state> shared_state = std::make_shared<pipes::shared_state>();
};

billiards::recognizer_t::recognizer_t()
    : impl_(std::make_unique<implementation>(*this))
{
}

billiards::recognizer_t::~recognizer_t() = default;

void billiards::recognizer_t::initialize()
{
    auto& m = *impl_;
    if (m.pipeline) { throw pipepp::pipe_exception("Recognizer already initialized"); }

    auto& pl = m.pipeline;
    pl = pipes::build_pipe();
    pl->launch();
}

void billiards::recognizer_t::destroy()
{
    auto& m = *impl_;
    if (m.pipeline == nullptr) { throw pipepp::pipe_exception("Recognizer already destroied"); }

    auto ref = std::move(m.pipeline);
    m.pipeline.reset();

    ref->sync();
}

void billiards::recognizer_t::refresh_image(frame_desc image, process_finish_callback_type&& callback)
{
    auto& m = *impl_;
    m.pipeline->suply(image, [&](pipes::shared_data& sty) {
        sty.callback = std::move(callback);
        sty.imdesc_bkup = image;
        sty.state = impl_->shared_state;
    });
}

std::weak_ptr<pipepp::detail::pipeline_base> billiards::recognizer_t::get_pipeline_instance() const
{
    return impl_->pipeline;
}

void billiards::recognizer_t::poll(std::unordered_map<std::string, cv::Mat>& shows)
{
}

billiards::recognition_desc const* billiards::recognizer_t::get_recognition() const
{
    return nullptr;
}

billiards::recognizer_t::frame_desc billiards::recognizer_t::get_image_snapshot() const
{
    return {};
}

std::vector<std::pair<std::string, std::chrono::microseconds>> billiards::recognizer_t::get_latest_timings() const
{
    return {};
}

namespace std
{
ostream& operator<<(ostream& strm, billiards::recognizer_t::frame_desc const& desc)
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

istream& operator>>(istream& strm, billiards::recognizer_t::frame_desc& desc)
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