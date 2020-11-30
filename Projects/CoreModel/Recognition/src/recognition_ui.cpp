#include <chrono>
#include <fstream>
#include <nana/gui.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <set>
#include <span>
#include "recognition.hpp"

#include "fmt/format.h"
#include "kangsw/atomic_queue.hxx"
#include "nana/gui/filebox.hpp"
#include "nana/gui/widgets/button.hpp"
#include "nana/gui/widgets/form.hpp"
#include "nana/gui/widgets/listbox.hpp"
#include "nana/gui/widgets/slider.hpp"
#include "nana/gui/widgets/textbox.hpp"
#include "nana/gui/widgets/treebox.hpp"
#include "nana/paint/graphics.hpp"
#include "nana/paint/pixel_buffer.hpp"
#include "pipepp/gui/basic_utility.hpp"
#include "pipepp/gui/option_panel.hpp"
#include "pipepp/gui/pipeline_board.hpp"
#include "pipepp/pipeline.hpp"
#include "tcp_server.hpp"

extern tcp_connection_desc     g_latest_conn;
extern billiards::recognizer_t g_recognizer;
static nlohmann::json          g_props;

using namespace std;

struct n_type {
    mutex                          shows_lock;
    unordered_map<string, cv::Mat> shows;
    nana::form                     fm{nana::API::make_center(800, 600)};
    nana::listbox                  lb{fm};
    map<string, nlohmann::json*>   param_mappings;
    atomic_bool                    dirty;
    struct {
        atomic_bool                      is_recording;
        shared_ptr<ostream>              strm_out;
        chrono::system_clock::time_point pivot_time;
        atomic_bool                      is_busy;
    } video;
};

struct video_frame {
    float                               time_point;
    billiards::recognizer_t::frame_desc img;
};

static ostream& operator<<(ostream& o, video_frame const& v)
{
    cv::Mat depth;
    cv::Mat rgb;
    cv::cvtColor(v.img.rgba, rgb, cv::COLOR_RGBA2RGB);
    v.img.depth.convertTo(depth, CV_8U, g_props["explorer"]["depth-alpha"]);

    auto wr = [&o](auto ty) { o.write((char*)&ty, sizeof ty); };
    wr(v.time_point);
    wr(v.img.camera);
    wr(v.img.camera_transform);
    wr(v.img.camera_translation);
    wr(v.img.camera_orientation);

    vector<uint8_t> buf;

    cv::imencode(".jpg", rgb, buf);
    wr((size_t)buf.size());
    o.write((char*)buf.data(), buf.size());

    buf.clear();
    cv::imencode(".jpg", depth, buf);
    wr((size_t)buf.size());
    o.write((char*)buf.data(), buf.size());

    return o;
}

static istream& operator>>(istream& i, video_frame& v)
{
    cv::Mat depth;
    cv::Mat rgb;

    auto rd = [&i](auto& ty) { i.read((char*)&ty, sizeof ty); };
    rd(v.time_point);
    rd(v.img.camera);
    rd(v.img.camera_transform);
    rd(v.img.camera_translation);
    rd(v.img.camera_orientation);

    vector<uint8_t> buf;

    size_t sz;
    rd(sz), buf.resize(sz);
    i.read((char*)buf.data(), sz);
    cv::cvtColor(cv::imdecode(buf, cv::IMREAD_COLOR), v.img.rgba, cv::COLOR_RGB2RGBA);

    rd(sz), buf.resize(sz);
    i.read((char*)buf.data(), sz);
    cv::imdecode(buf, cv::IMREAD_GRAYSCALE).convertTo(v.img.depth, CV_32F, 1.f / (float)g_props["explorer"]["depth-alpha"]);

    return i;
}

struct video_frame_chunk {
    float time_point;
    struct {
        cv::Matx44f                                camera_transform;
        billiards::recognizer_t::camera_param_type camera;
        cv::Vec3f                                  camera_translation;
        cv::Vec4f                                  camera_orientation;
    } img;
    vector<uint8_t> chnk_rgb;
    vector<uint8_t> chnk_depth;
};

static istream& operator>>(istream& i, video_frame_chunk& v)
{
    cv::Mat depth;
    cv::Mat rgb;

    auto rd = [&i](auto& ty) { i.read((char*)&ty, sizeof ty); };
    rd(v.time_point);
    rd(v.img.camera);
    rd(v.img.camera_transform);
    rd(v.img.camera_translation);
    rd(v.img.camera_orientation);

    vector<uint8_t> buf;

    size_t sz;
    rd(sz), v.chnk_rgb.resize(sz);
    i.read((char*)v.chnk_rgb.data(), sz);

    rd(sz), v.chnk_depth.resize(sz);
    i.read((char*)v.chnk_depth.data(), sz);

    return i;
}

static video_frame parse(video_frame_chunk const& va)
{
    video_frame v;
    v.img.camera_orientation = va.img.camera_orientation;
    v.img.camera             = va.img.camera;
    v.img.camera_transform   = va.img.camera_transform;
    v.img.camera_translation = va.img.camera_translation;
    v.time_point             = va.time_point;

    cv::cvtColor(cv::imdecode(va.chnk_rgb, cv::IMREAD_COLOR), v.img.rgba, cv::COLOR_RGB2RGBA);
    cv::imdecode(va.chnk_depth, cv::IMREAD_GRAYSCALE).convertTo(v.img.depth, CV_32F, 1.f / (float)g_props["explorer"]["depth-alpha"]);

    return v;
}

static weak_ptr<n_type> n_weak;

static string getImgType(int imgTypeInt)
{
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] = {CV_8U, CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8S, CV_8SC1, CV_8SC2, CV_8SC3, CV_8SC4, CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4, CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4, CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    string enum_strings[] = {"8U", "8UC1", "8UC2", "8UC3", "8UC4", "8S", "8SC1", "8SC2", "8SC3", "8SC4", "16U", "16UC1", "16UC2", "16UC3", "16UC4", "16S", "16SC1", "16SC2", "16SC3", "16SC4", "32S", "32SC1", "32SC2", "32SC3", "32SC4", "32F", "32FC1", "32FC2", "32FC3", "32FC4", "64F", "64FC1", "64FC2", "64FC3", "64FC4"};

    for (int i = 0; i < numImgTypes; i++) {
        if (imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}

struct mat_desc_row_t {
    optional<bool>                   is_displaying = false;
    string                           mat_name;
    cv::Mat                          mat;
    chrono::system_clock::time_point tp;
};

string time_to_from_string(chrono::system_clock::time_point tp)
{
    using namespace chrono;
    auto now = chrono::system_clock::now();
    auto gap = duration_cast<seconds>(now - tp);

    auto seconds = gap.count() % 60;
    auto minutes = gap.count() / 60 % 60;
    auto hours   = gap.count() / (60 * 60) % 24;
    auto days    = gap.count() / (60 * 60 * 24) % 7;
    auto weeks   = gap.count() / (60 * 60 * 24 * 7) % 4;
    auto months  = gap.count() / (60 * 60 * 24 * 7 * 4) % 12;
    auto years   = gap.count() / (60 * 60 * 24 * 7 * 4 * 12);

    long long values[] = {years, months, weeks, days, hours, minutes, seconds};
    string    tags[]   = {"year", "month", "week", "day", "hour", "minute", "second"};

    for (int i = 0; i < _countof(values); ++i) {
        if (values[i]) {
            return to_string(values[i]) + " " + tags[i] + (values[i] > 1 ? "s ago" : " ago");
        }
    }

    return "now";
}

nana::listbox::oresolver& operator<<(nana::listbox::oresolver& ores, const mat_desc_row_t const& data)
{
    ores << data.mat_name;
    ores << data.mat.cols;
    ores << data.mat.rows;
    ores << getImgType(data.mat.type());
    ores << time_to_from_string(data.tp);

    return ores;
}

using nlohmann::json;

static void json_recursive_substitute(json& to, json const& from)
{
    int index = 0;
    for (auto& pair : to.items()) {
        auto&       value = pair.value();
        json const* src   = nullptr;
        {
            cout << "info: subtitute key " << pair.key();
            if (auto it = from.find(pair.key()); it != from.end()) {
                cout << " success";
                src = &*it;
            }
            cout << endl;
        }

        if (src) {
            if (value.type() == json::value_t::object) {
                json_recursive_substitute(value, *src);
            } else if (value.type() == json::value_t::array) {
                for (int i = 0; i < /*min(value.size(),*/ (src->size()); ++i) {
                    if (value[i].type() == nlohmann::detail::value_t::object) {
                        json_recursive_substitute(value[i], (*src)[i]);
                    } else {
                        value[i] = (*src)[i];
                    }
                }
            } else if (strcmp(value.type_name(), src->type_name()) == 0) {
                value = *src;
            }
        }
    }
}

static const auto recognizer_handler = [](auto&& imdsc, nlohmann::json const& data) {
    if (auto conn = g_latest_conn.socket.lock()) {
        auto p_data = make_shared<string>(data.dump());
        p_data->push_back('\n');
        conn->async_write_some(
          boost::asio::const_buffer{p_data->c_str(), p_data->size()},
          [p_data](auto&&, auto&&) {});
    }

    ui_on_refresh(&imdsc);
};

void ui_on_refresh(billiards::recognizer_t::frame_desc const* fd);
void exec_ui()
{
    using namespace nana;
    auto n   = make_shared<n_type>();
    n_weak   = n;
    form& fm = n->fm; // 주 폼

    // 변수 목록
    string AUTOSAVE_PATH = "arbilliards-autosave.json";
    AUTOSAVE_PATH        = (filesystem::current_path() / AUTOSAVE_PATH).string();
    cout << "Autosave path set at " << AUTOSAVE_PATH << endl;

    string current_save_path = AUTOSAVE_PATH;
    string curtime_prefix;
    {
        time_t rtime;
        tm     t;
        char   buff[128];
        time(&rtime);
        localtime_s(&t, &rtime);
        strftime(buff, sizeof buff, "%Y-%m-%d %H %M %S", &t);
        curtime_prefix = buff;
    }
    bool is_config_dirty = false;

    // -- 상태 창
    optional<string> state_messages;

    // 타이틀바 창 업데이트 함수
    auto fm_caption_dirty = [&]() {
        auto str    = "AR Billiards Image Processing Core - "s;
        auto divide = current_save_path.find_last_of("\\");
        str += divide == npos ? current_save_path : current_save_path.substr(divide + 1);
        if (is_config_dirty) {
            str += " (*)";
        }
        fm.caption(str);
    };
    fm_caption_dirty();

    // 파일 로드 함수
    auto load_from_path = [&](string path) {
        if (ifstream strm(path); strm.is_open()) {
            try {
                json parsed = json::parse((stringstream() << strm.rdbuf()).str());
                g_recognizer.get_pipeline_instance().lock()->import_options(parsed);

                // 윈도우 위치도 로드
                // 에디터 설정 목록
                g_props = parsed["g_props"];
                if (g_props.contains("window-position")) {
                    array<int, 4> wndPos = g_props.at("window-position");
                    fm.move((rectangle&)wndPos);
                }

                g_props["explorer"]["depth-alpha"] = 32;

                current_save_path = path;
                is_config_dirty   = false;
                state_messages.emplace("loaded configurations from file "s + path);
                fm_caption_dirty();

                return true;
            } catch (std::exception& e) {
                cout << "Failed to load configuration file ... " << endl;
                cout << e.what() << endl;
            }
        }
        return false;
    };

    // 세이브 함수
    auto save_as = [&](string path) {
        ofstream strm(path);
        json     opts;
        g_recognizer.get_pipeline_instance().lock()->export_options(opts);
        opts["g_props"] = g_props;
        strm << opts.dump(4);

        is_config_dirty = false;
        state_messages.emplace("saved configurations to file "s + path);
        fm_caption_dirty();
    };

    // 설정 로드
    if (load_from_path(AUTOSAVE_PATH)) {}
    pipepp::gui::option_panel options{fm, true};
    options.on_dirty = [&](auto&&) { is_config_dirty = true, fm_caption_dirty(); };
    options.vertical(true);

    const auto reload_global_opts = [&]() {
        options.reload(g_recognizer.get_pipeline_instance(), &g_recognizer.get_pipeline_instance().lock()->options());
    };
    reload_global_opts();

    // -- 리셋, 익스포트, 임포트 버튼 구현
    button btn_reset(fm), btn_export(fm), btn_import(fm);
    btn_reset.caption("Reset");
    btn_export.caption("Export Config ...(Alt+Q)");
    btn_import.caption("Import Config ...(Alt+O)");

    btn_reset.bgcolor(colors::gray);
    btn_reset.events().click([&](arg_click const& arg) {
        msgbox mb(arg.window_handle, "Parameter Reset", msgbox::yes_no);
        mb.icon(msgbox::icon_question);
        mb << "Are you sure?";

        if (mb.show() == msgbox::pick_yes) {
            //g_props = billiards::recognizer_t().get_props();
            reload_global_opts();
        }
    });
    btn_export.events().click([&](arg_click const& arg) {
        filebox fb(fm, false);
        fb.add_filter("Json File", "*.json");
        fb.add_filter("All", "*.*");
        fb.allow_multi_select(false);

        if (auto paths = fb(); !paths.empty()) {
            auto path = paths.front();
            save_as(path.string());
            current_save_path = path.string();
            fm_caption_dirty();
        }
    });
    btn_import.events().click([&](arg_click const& arg) {
        filebox fb(fm, true);
        fb.add_filter("Json File", "*.json");
        fb.add_filter("All", "*.*");
        fb.allow_multi_select(false);

        if (auto paths = fb(); !paths.empty()) {
            auto path = paths.front();
            auto res  = load_from_path(path.string());
            reload_global_opts();
        }
    });

    // -- 표시되는 이미지 목록
    auto& matlist = n->lb;
    matlist.append_header("Mat Name", 160);
    matlist.append_header("W", 40);
    matlist.append_header("H", 40);
    matlist.append_header("Type", 50);
    matlist.append_header("Updated");
    matlist.sort_col(0);

    // 이벤트 바인딩
    matlist.checkable(true);
    matlist.events().checked([&](arg_listbox const& arg) {
        auto ptr = arg.item.value_ptr<mat_desc_row_t>();

        if (ptr) {
            arg.item.value<mat_desc_row_t>().is_displaying = arg.item.checked();
            n->dirty                                       = true;
        }
    });

    // 이미지 목록
    timer list_update_timer(333ms);
    list_update_timer.elapse([&]() {
        if (!fm.enabled()) {
            list_update_timer.stop();
            return;
        }

        matlist.auto_draw(false);
        for (auto item_proxy : matlist.at(0)) {
            try {
                auto& val = item_proxy.value<mat_desc_row_t>();
                item_proxy.resolve_from(val);
            } catch (exception e) {
                cout << e.what() << endl;
            }
        }
        matlist.auto_draw(true);
    });
    list_update_timer.start();

    // -- 틱 미터 박스
    listbox tickmeters(fm);
    tickmeters.append_header("", 20);
    tickmeters.append_header("Name", 160);
    tickmeters.append_header("Elapsed", 240);
    tickmeters.sort_col(0);

    tickmeters.set_sort_compare(0, [](string str1, nana::any* a, string str2, nana::any* b, bool reverse) {
        if (reverse) {
            swap(str1, str2);
        }

        if (str1 == "-") {
            return false;
        }
        if (str2 == "-") {
            return true;
        }

        int s1 = stoi(str1);
        int s2 = stoi(str2);

        return s1 < s2;
    });

    // -- 파이프라인
    pipepp::gui::DEFAULT_DATA_FONT = paint::font{"consolas", 10.5};
    kangsw::atomic_queue<std::pair<std::string, cv::Mat>> shown_images{1024};
    kangsw::atomic_queue<std::string>                     shutdown_images{1024};
    pipepp::gui::pipeline_board                           pl_board(fm, {}, true);
    pl_board.reset_pipeline(g_recognizer.get_pipeline_instance().lock());
    pl_board.bgcolor(colors::antique_white);
    // pl_board.main_connection_line_color = color(255, 255, 255);
    // pl_board.optional_connection_line_color = color(0, 255, 255);

    pl_board.events().mouse_down([&](arg_mouse const& arg) { if(arg.mid_button) { pl_board.center();} });
    pl_board.debug_data_subscriber = [&](string_view basic_string, pipepp::execution_context_data::debug_data_entity const& debug_data_entity) {
        if (auto any_ptr = std::get_if<std::any>(&debug_data_entity.data)) {
            if (auto mat_ptr = std::any_cast<cv::Mat>(any_ptr)) {
                decltype(shown_images)::element_type e;
                e.first  = fmt::format("{0}/{1}", basic_string, debug_data_entity.name);
                e.second = *mat_ptr;
                if (mat_ptr->channels() == 3) {
                    cv::cvtColor(*mat_ptr, *mat_ptr, cv::COLOR_RGB2BGR);
                }
                shown_images.try_push(std::move(e));
                return true;
            }
        }
        return false;
    };
    pl_board.option_changed = [&](pipepp::pipe_id_t pipe_id, string_view basic_string_view) {
        is_config_dirty = true;
        fm_caption_dirty();
    };
    pl_board.debug_data_unchecked = [&](string_view basic_string, pipepp::execution_context_data::debug_data_entity const& debug_data_entity) {
        if (auto any_ptr = std::get_if<std::any>(&debug_data_entity.data)) {
            if (auto mat_ptr = std::any_cast<cv::Mat>(any_ptr)) {
                auto name = fmt::format("{0}/{1}", basic_string, debug_data_entity.name);
                shutdown_images.try_push(move(name));
            }
        }
    };

    // -- Waitkey 폴링 타이머
    timer tm_waitkey{16ms};
    tm_waitkey.elapse([&]() {
        cv::waitKey(1);

        if (!fm.enabled()) {
            tm_waitkey.stop();
            return;
        }

        pl_board.update();

        if (false) {
            std::cout << "\r";
            auto  pipe = g_recognizer.get_pipeline_instance().lock();
            auto& pool = pipe->_thread_pool();
            using std::chrono::duration;

            fmt::print("{:<80}",
                       fmt::format("n_threads: [{0:>3}/{1:>3}], task_wait: {2:>10.4f}ms, task_interv: {3:>10.4f}ms",
                                   pool.num_available_workers(),
                                   pool.num_workers(),
                                   duration<double, milli>(pool.average_wait()).count(),
                                   duration<double, milli>(pool.average_interval()).count()));
        }

        for (std::pair<std::string, cv::Mat> fetch_data;
             shown_images.try_pop(fetch_data);) {
            cv::imshow(fetch_data.first, fetch_data.second);
        }

        for (std::string fetch_str; shutdown_images.try_pop(fetch_str);) {
            cv::destroyWindow(fetch_str);
        }
    });
    tm_waitkey.start();

    // -- 스냅샷 관련
    optional<billiards::recognizer_t::frame_desc> snapshot;
    timer                                         snapshot_loader{100ms};
    button                                        btn_snap_load(fm), btn_snapshot(fm), btn_snap_abort(fm);
    btn_snap_load.caption("Load Snapshot (Alt+E)");
    btn_snapshot.caption("Capture Snapshot (Alt+C)");
    btn_snap_abort.caption("Abort Snapshot");

    btn_snap_abort.events().click([&](auto) { snapshot.reset(); });
    btn_snap_load.events().click([&](auto) {
        filebox fb(fm, true);
        fb.add_filter("Ar Billiards Snapshot Format", "*.arbsnap");
        fb.allow_multi_select(false);

        if (auto paths = fb(); paths.empty() == false) {
            auto     path = paths.front().string();
            ifstream strm{path, ios::binary | ios::in};

            if (strm.is_open()) {
                strm >> snapshot.emplace();
                snapshot_loader.start();
            }
        }
    });
    btn_snapshot.events().click([&](auto) {
        auto    snap = g_recognizer.get_image_snapshot();
        filebox fb(fm, false);

        for (int i = 0; i < 100; ++i) {
            auto start_path_str = fb.path().string();
            start_path_str.pop_back();
            auto file_name  = curtime_prefix + " ("s + to_string(i) + ").arbsnap"s;
            auto start_path = start_path_str / filesystem::path(file_name);
            if (!filesystem::exists(start_path)) {
                fb.init_file(file_name);
                break;
            }
        }

        fb.add_filter("Ar Billiards Snapshot Format", "*.arbsnap");
        fb.allow_multi_select(false);

        if (auto paths = fb(); paths.empty() == false) {
            auto     path = paths.front().string();
            ofstream strm{path, ios::binary | ios::out};
            strm << snap;
        }
    });
    snapshot_loader.elapse([&]() {
        if (snapshot) {
            g_recognizer.refresh_image(*snapshot, recognizer_handler);
        } else {
            snapshot_loader.stop();
        }
    });

    // -- 로딩
    button                    btn_video_load(fm), btn_video_record(fm), btn_video_playpause(fm);
    vector<video_frame_chunk> frame_chunks;
    optional<video_frame>     previous;
    slider                    video_slider(fm);
    bool                      is_playing_video = false;

    btn_video_load.caption("Load Video");
    btn_video_record.caption("Record Video");
    btn_video_playpause.caption("Play");
    video_slider.seek(slider::seekdir::bilateral);
    video_slider.vertical(false);
    video_slider.bgcolor(colors::light_blue);
    timer video_player;

    btn_video_record.events().click([&](auto) {
        if (is_playing_video) {
            msgbox("Pause playing video before record!");
            return;
        }

        if (n->video.is_recording) {
            n->video.is_recording = false;
            btn_video_record.caption("Record Video");
            btn_video_record.bgcolor(colors::light_gray);
            return;
        }

        filebox fb(fm, false);
        fb.add_filter("AR Billiards Video Trace Type", "*.arbtrace");
        fb.allow_multi_select(false);

        if (auto paths = fb.show(); !paths.empty()) {
            auto& path            = paths.front();
            n->video.strm_out     = make_shared<ofstream>(path.string(), ios::out | ios::binary);
            n->video.pivot_time   = chrono::system_clock::now();
            n->video.is_recording = true;
            btn_video_record.caption("Stop Recording");
            btn_video_record.bgcolor(colors::red);
        }
    });
    btn_video_load.events().click([&](auto) {
        if (frame_chunks.empty() == false) {
            is_playing_video = true;
            btn_video_playpause.events().click.emit({}, fm);
            frame_chunks.clear();
            btn_video_load.bgcolor(colors::light_gray);
            btn_video_load.caption("Load Video");
            return;
        }

        filebox fb(fm, true);
        fb.add_filter("AR Billiards Video Trace Type", "*.arbtrace");
        fb.allow_multi_select(false);

        if (auto paths = fb.show(); !paths.empty()) {
            auto path = paths.front().string();
            frame_chunks.clear();

            ifstream in{path, ios::in | ios::binary};
            while (!in.eof()) {
                in >> frame_chunks.emplace_back();
            }
            video_player.interval(10ms);
            video_player.start();

            btn_video_load.bgcolor(colors::forest_green);
            btn_video_load.caption("Unload Video");
        }
    });
    btn_video_playpause.events().click([&](auto) {
        is_playing_video = !is_playing_video;
        if (is_playing_video && frame_chunks.empty()) {
            is_playing_video = false;
        }

        if (is_playing_video) {
            btn_video_playpause.bgcolor(colors::red);
            btn_video_playpause.caption("Pause");
        } else {
            btn_video_playpause.bgcolor(colors::light_gray);
            btn_video_playpause.caption("Play");
        }
    });
    video_player.elapse([&]() {
        if (frame_chunks.empty() == false) {
            if (is_playing_video) {
                video_slider.move_step(false);
            } else {
                video_slider.events().value_changed.emit({video_slider}, fm);
            }
        } else {
            video_slider.borderless(true);
            video_player.stop();
            previous.reset();
        }
    });
    video_slider.events().value_changed([&](arg_slider v) {
        if (!frame_chunks.empty()) {
            if (video_slider.maximum() != frame_chunks.size()) {
                video_slider.maximum(frame_chunks.size());
            }
            auto& vid_chnk = frame_chunks[video_slider.value() % frame_chunks.size()];
            auto  vid      = parse(vid_chnk);
            if (previous) {
                n->video.is_busy = true;
                auto tm          = clamp(vid.time_point - previous->time_point, 0.001f, 0.1f) * 0.75f;

                g_recognizer.refresh_image(previous->img, recognizer_handler);

                video_player.interval(chrono::milliseconds((int)(is_playing_video ? tm * 1000.f : 100)));
            }
            previous = vid;
        }
    });

    // -- 단축키 집합
    fm.register_shortkey(L's');
    fm.register_shortkey(L'o');
    fm.register_shortkey(L'e');
    fm.register_shortkey(L'q');
    fm.register_shortkey(L'c');
    fm.register_shortkey(L'a');
    fm.events().shortkey([&](arg_keyboard key) {
        switch (key.key) {
            case L'q':
                btn_export.events().click.emit({}, fm);
                break;
            case L's':
                save_as(current_save_path);
                break;
            case L'o':
                btn_import.events().click.emit({}, fm);
                break;
            case L'e':
                btn_snap_load.events().click.emit({}, fm);
                break;
            case L'c':
                btn_snapshot.events().click.emit({}, fm);
                break;
            case L'a':
                btn_video_playpause.events().click.emit({}, fm);
                break;
        }
    });

    // -- 종료 쿼리
    fm.events().unload([&](arg_unload const& ul) {
        if (is_config_dirty) {
            msgbox quitbox(fm, "Quit", msgbox::yes_no_cancel);
            quitbox.icon(msgbox::icon_question);

            quitbox << "Save changes before exit";

            switch (quitbox.show()) {
                case msgbox::pick_ok:
                case msgbox::pick_yes:
                    save_as(current_save_path);
                    break;
                case msgbox::pick_no:
                    break;
                case msgbox::pick_cancel:
                    ul.cancel = true;
                    break;
                default:;
            }
        }
    });

    // -- 레이아웃 설정
    place layout(fm);
    // <vert
    //   <mat_lists>
    //   <timings>
    // >
    // <options margin = 5>
    //   <enter weight = 30 margin = 5>
    auto layout_divider_string = R"(
      <vert
        <
          <center margin=5>
          <vert
            weight=400
            <margin=[5,5,2,5] gap=5 weight=30 <btn_export><weight=10><btn_import>>
            <options margin=[5,5,2,5]>
            <margin=[0,5,5,5] gap=5 weight=30 <btn_video_load><btn_video_record><btn_video_playpause>>
          >
        >
        <
          weight=60 margin=[15,5,10,5]
          <video_slider>
        >
      >)";
    layout.div(layout_divider_string);

    // DISCARDED
    // "    <margin=[0,5,5,5] gap=5 weight=30 <btn_snap_load><btn_snapshot><btn_snap_abort weight=25%>>"

    layout["center"] << pl_board;
    layout["options"] << options;
    layout["btn_reset"] << btn_reset;
    layout["btn_export"] << btn_export;
    layout["btn_import"] << btn_import;
    layout["mat_lists"] << matlist;
    layout["timings"] << tickmeters;
    layout["btn_snap_load"] << btn_snap_load;
    layout["btn_snapshot"] << btn_snapshot;
    layout["btn_snap_abort"] << btn_snap_abort;
    layout["btn_video_load"] << btn_video_load;
    layout["btn_video_record"] << btn_video_record;
    layout["btn_video_playpause"] << btn_video_playpause;
    layout["video_slider"] << video_slider;
    layout.collocate();

    fm.events().move([&](auto) {
        auto rect                  = rectangle(fm.pos(), fm.size());
        g_props["window-position"] = (array<int, 4>&)rect;
    });
    fm.events().resized([&](auto) {
        auto rect                  = rectangle(fm.pos(), fm.size());
        g_props["window-position"] = (array<int, 4>&)rect;
        static_assert(sizeof rect == sizeof(array<int, 4>));
    });

    fm.show();
    exec();

    save_as(AUTOSAVE_PATH);

    this_thread::sleep_for(100ms);
    cout << "info: GUI Expired" << endl;
    cv::destroyAllWindows();
    cv::waitKey(1);

    n.reset();
}

void ui_on_refresh(billiards::recognizer_t::frame_desc const* fd)
{
    if (auto n = n_weak.lock(); n) {
        {
            lock_guard<mutex> lock{n->shows_lock};
            n->shows.clear();
            g_recognizer.poll(n->shows);
        }

        if (n->video.is_recording) {
            assert(n->video.strm_out);
            auto        time = chrono::duration<float>(chrono::system_clock::now() - n->video.pivot_time).count();
            video_frame f    = {.time_point = time, .img = *fd};
            *n->video.strm_out << f;
        } else if (n->video.strm_out) {
            n->video.strm_out.reset();
        }

        n->video.is_busy = false;
        n->dirty         = true;
    }
}
