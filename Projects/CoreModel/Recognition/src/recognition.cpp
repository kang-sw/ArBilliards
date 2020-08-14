#include "recognition.hpp"
#include <thread>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <condition_variable>
#include <iostream>
#include <map>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

namespace billiards
{
using img_t = recognizer_t::parameter_type;
using img_cb_t = recognizer_t::process_finish_callback_type;
using opt_img_t = optional<img_t>;
using read_lock = shared_lock<shared_mutex>;
using write_lock = unique_lock<shared_mutex>;
class recognizer_impl_t
{
public:
public:
    thread worker;
    atomic_bool worker_is_alive;
    condition_variable_any worker_event_wait;
    mutex worker_event_wait_mtx;

    opt_img_t img_cue;
    shared_mutex img_cue_mtx;
    img_cb_t img_cue_cb;

    map<string, cv::Mat> img_show;
    shared_mutex img_show_mtx;

public:
    recognizer_impl_t()
    {
        worker_is_alive = true;
        worker = thread(&recognizer_impl_t::async_worker_thread, this);
    }

    ~recognizer_impl_t()
    {
        if (worker.joinable()) {
            worker_is_alive = false;
            worker_event_wait.notify_all();
            worker.join();
        }
    }

    void show_image(string wnd_name, cv::Mat img)
    {
        write_lock lock(img_show_mtx);
        img_show[move(wnd_name)] = move(img);
    }

    void async_worker_thread()
    {
        while (worker_is_alive) {
            {
                unique_lock<mutex> lck(worker_event_wait_mtx);
                worker_event_wait.wait(lck);
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
                auto desc = proc_img(*img);
                if (on_finish) { on_finish(*img, desc); }
            }
        }
    }

    recognition_desc proc_img(img_t const& img)
    {
        // TODO: 이미지 처리하기
        show_image("default", img.rgb);
        return {};
    }
};

recognizer_t::recognizer_t()
    : impl_(make_unique<recognizer_impl_t>())
{
}

recognizer_t::~recognizer_t() = default;

void recognizer_t::refresh_image(parameter_type image, process_finish_callback_type callback)
{
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

void recognizer_t::poll()
{
    auto& m = *impl_;
    decltype(m.img_show) shows;

    if (read_lock lock(m.img_show_mtx, try_to_lock); lock) {
        shows = move(m.img_show);
        m.img_show = {};
    }

    for (auto& pair : shows) {
        imshow(pair.first, pair.second);
    }
}
} // namespace billiards