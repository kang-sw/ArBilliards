#include "recognition.hpp"
#include <set>
#include <fstream>
#include <chrono>
#include <nana/gui.hpp>
#include <queue>
#include <opencv2/opencv.hpp>
#include <span>

extern billiards::recognizer_t g_recognizer;

using namespace std;

#include <nana/gui/widgets/form.hpp>
#include <nana/gui/widgets/scroll.hpp>
#include <nana/gui/widgets/treebox.hpp>
#include <nana/gui/widgets/textbox.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/widgets/listbox.hpp>
#include <nana/gui/filebox.hpp>

struct n_type {
    mutex shows_lock;
    unordered_map<string, cv::Mat> shows;
    nana::form fm{nana::API::make_center(800, 600)};
    nana::listbox lb{fm};
    map<string, nlohmann::json*> param_mappings;
    atomic_bool dirty;
    struct {
        atomic_bool is_recording;
        shared_ptr<ostream> strm_out;
        chrono::system_clock::time_point pivot_time;
        atomic_bool is_busy;
    } video;
};

struct video_frame {
    float time_point;
    billiards::recognizer_t::parameter_type img;
};

#define DEPTH_RATE 32

static ostream& operator<<(ostream& o, video_frame const& v)
{
    cv::Mat depth;
    cv::Mat rgb;
    cv::cvtColor(v.img.rgba, rgb, cv::COLOR_RGBA2RGB);
    v.img.depth.convertTo(depth, CV_8U, DEPTH_RATE);

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
    cv::imdecode(buf, cv::IMREAD_GRAYSCALE).convertTo(v.img.depth, CV_32F, 1.f / DEPTH_RATE);

    return i;
}

struct video_frame_chunk {
    float time_point;
    struct {
        cv::Matx44f camera_transform;
        billiards::recognizer_t::camera_param_type camera;
        cv::Vec3f camera_translation;
        cv::Vec4f camera_orientation;
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
    v.img.camera = va.img.camera;
    v.img.camera_transform = va.img.camera_transform;
    v.img.camera_translation = va.img.camera_translation;
    v.time_point = va.time_point;

    cv::cvtColor(cv::imdecode(va.chnk_rgb, cv::IMREAD_COLOR), v.img.rgba, cv::COLOR_RGB2RGBA);
    cv::imdecode(va.chnk_depth, cv::IMREAD_GRAYSCALE).convertTo(v.img.depth, CV_32F, 1.f / DEPTH_RATE);

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
    optional<bool> is_displaying = false;
    string mat_name;
    cv::Mat mat;
    chrono::system_clock::time_point tp;
};

string time_to_from_string(chrono::system_clock::time_point tp)
{
    using namespace chrono;
    auto now = chrono::system_clock::now();
    auto gap = duration_cast<seconds>(now - tp);

    auto seconds = gap.count() % 60;
    auto minutes = gap.count() / 60 % 60;
    auto hours = gap.count() / (60 * 60) % 24;
    auto days = gap.count() / (60 * 60 * 24) % 7;
    auto weeks = gap.count() / (60 * 60 * 24 * 7) % 4;
    auto months = gap.count() / (60 * 60 * 24 * 7 * 4) % 12;
    auto years = gap.count() / (60 * 60 * 24 * 7 * 4 * 12);

    long long values[] = {years, months, weeks, days, hours, minutes, seconds};
    string tags[] = {"year", "month", "week", "day", "hour", "minute", "second"};

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

static void json_iterative_substitute(json& to, json const& from)
{
    int index = 0;
    for (auto& pair : to.items()) {
        auto& value = pair.value();
        json const* src = nullptr;
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
                json_iterative_substitute(value, *src);
            }
            else if (value.type() == json::value_t::array) {
                for (int i = 0; i < min(value.size(), src->size()); ++i) {
                    if (value[i].type() == nlohmann::detail::value_t::object) {
                        json_iterative_substitute(value[i], (*src)[i]);
                    }
                    else {
                        value[i] = (*src)[i];
                    }
                }
            }
            else if (strcmp(value.type_name(), src->type_name()) == 0) {
                value = *src;
            }
        }
    }
}

void exec_ui()
{
    using namespace nana;
    auto n = make_shared<n_type>();
    n_weak = n;
    form& fm = n->fm; // 주 폼

    // 변수 목록
    string AUTOSAVE_PATH = "arbilliards-autosave.json";
    AUTOSAVE_PATH = (filesystem::current_path() / AUTOSAVE_PATH).string();
    cout << "Autosave path set at " << AUTOSAVE_PATH << endl;

    string current_save_path = AUTOSAVE_PATH;
    string curtime_prefix;
    {
        time_t rtime;
        tm t;
        char buff[128];
        time(&rtime);
        localtime_s(&t, &rtime);
        strftime(buff, sizeof buff, "%Y-%m-%d %H %M %S", &t);
        curtime_prefix = buff;
    }
    bool is_config_dirty = false;

    // -- 상태 창
    queue<string> state_messages;

    // 타이틀바 창 업데이트 함수
    auto fm_caption_dirty = [&]() {
        auto str = "AR Billiards Image Processing Core - "s;
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

                // 윈도우 위치도 로드
                if (auto it = parsed.find("window-position"); it != parsed.end()) {
                    array<int, 4> wndPos = *it;
                    fm.move((rectangle&)wndPos);
                }

                json_iterative_substitute(g_recognizer.props, parsed);
                current_save_path = path;
                is_config_dirty = false;
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
        strm << g_recognizer.props.dump(4);

        is_config_dirty = false;
        state_messages.emplace("saved configurations to file "s + path);
        fm_caption_dirty();
    };

    // 설정 로드
    if (load_from_path(AUTOSAVE_PATH)) { }

    treebox tr(fm);
    textbox param_enter_box(fm);
    optional<treebox::item_proxy> selected_proxy;
    param_enter_box.multi_lines(false);

    // -- JSON 파라미터 입력 창 핸들
    auto param_enter_query = [&](bool apply_change) {
        if (selected_proxy) {
            if (auto it = n->param_mappings.find(selected_proxy->key()); it != n->param_mappings.end()) {
                auto text = param_enter_box.text();
                auto prop = *it->second;
                auto original_type = prop.type_name();
                prop = json::parse(text, nullptr, false);

                if (strcmp(original_type, prop.type_name()) != 0) { // 파싱에 실패하면, 아무것도 하지 않습니다.
                    param_enter_box.bgcolor(colors::light_pink);
                    cout << "error: type is " << prop.type_name() << endl;
                    return;
                }

                // 파싱에 성공했다면 즉시 이를 반영합니다.
                param_enter_box.bgcolor(colors::light_green);

                if (apply_change) {
                    *it->second = prop;

                    auto new_label = selected_proxy.value().text();
                    new_label = new_label.substr(0, new_label.find(' '));
                    new_label += "  [" + param_enter_box.text() + ']';

                    selected_proxy->text(new_label);
                    param_enter_box.select(true);

                    drawing(tr).update();

                    is_config_dirty = true;
                    fm_caption_dirty();
                }
            }
        }
        else {
            param_enter_box.bgcolor(colors::light_gray);
        }
    };

    param_enter_box.events().key_char([&](arg_keyboard const& arg) {
        if (arg.key == 13) {
            param_enter_query(true);
        }
    });

    param_enter_box.events().text_changed([&](arg_textbox const& arg) { param_enter_query(false); });

    // -- JSON 파라미터 트리 빌드
    struct node_iterative_constr_t {
        static void exec(treebox& tree, treebox::item_proxy root, json const& elem)
        {
            auto n = n_weak.lock();
            for (auto& prop : elem.items()) {
                string value_text = prop.key();
                string key = root.key() + prop.key();

                n->param_mappings[key] = const_cast<json*>(&prop.value());

                if (prop.value().is_object() || prop.value().is_array()) {
                    auto node = tree.insert(root, key, move(value_text));

                    exec(tree, node, prop.value());
                }
                else {
                    value_text += "  [" + prop.value().dump() + ']';
                    tree.insert(root, key, move(value_text));
                }
            }
        }
    };

    auto reload_tr = [&]() {
        tr.clear();
        node_iterative_constr_t::exec(tr, tr.insert("param", "Parameters").expand(true), g_recognizer.props);

        // -- JSON 파라미터 선택 처리
        tr.events().selected([&](arg_treebox const& arg) {
            // 말단 노드인 경우에만 입력 창을 활성화합니다.
            if (!arg.operated) {
                return;
            }

            if (arg.item.child().empty()) {
                if (auto it = n->param_mappings.find(arg.item.key());
                    arg.operated && it != n->param_mappings.end()) {
                    selected_proxy = arg.item;
                    auto selected = *it;

                    param_enter_box.select(true);
                    param_enter_box.del();
                    param_enter_box.append(selected.second->dump(), true);
                    param_enter_box.select(true);
                    param_enter_box.clear_undo();
                    param_enter_box.focus();
                    return;
                }
            }
            else { // 말단 노드가 아니라면 expand
                arg.item.expand(true);
            }
            selected_proxy = {};
            param_enter_box.select(true), param_enter_box.del();
        });
    };
    reload_tr();

    // -- 리셋, 익스포트, 임포트 버튼 구현
    button btn_reset(fm), btn_export(fm), btn_import(fm);
    btn_reset.caption("Reset");
    btn_export.caption("Export As...(Alt+Q)");
    btn_import.caption("Import From...(Alt+O)");

    btn_reset.bgcolor(colors::gray);
    btn_reset.events().click([&](arg_click const& arg) {
        msgbox mb(arg.window_handle, "Parameter Reset", msgbox::yes_no);
        mb.icon(msgbox::icon_question);
        mb << "Are you sure?";

        if (mb.show() == msgbox::pick_yes) {
            g_recognizer.props = billiards::recognizer_t().props;
            reload_tr();
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
        }
    });
    btn_import.events().click([&](arg_click const& arg) {
        filebox fb(fm, true);
        fb.add_filter("Json File", "*.json");
        fb.add_filter("All", "*.*");
        fb.allow_multi_select(false);

        if (auto paths = fb(); !paths.empty()) {
            auto path = paths.front();
            auto res = load_from_path(path.string());
            reload_tr();
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
            n->dirty = true;
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

    tickmeters.set_sort_compare(0, [](string str1, any* a, string str2, any* b, bool reverse) {
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

    // -- Waitkey 폴링 타이머
    timer tm_waitkey{16ms};
    tm_waitkey.elapse([&]() {
        cv::waitKey(1);

        if (!fm.enabled()) {
            tm_waitkey.stop();
            return;
        }

        if (n->dirty) {
            // 이미지 목록을 업데이트
            decltype(n->shows) shows;
            {
                lock_guard<mutex> lock{n->shows_lock};
                shows = move(n->shows);
            }
            n->dirty = false;

            matlist.auto_draw(false);
            for (auto& pair : shows) {
                bool existing_key = false;

                for (auto item_proxy : matlist.at(0)) {
                    auto& val = item_proxy.value<mat_desc_row_t>();
                    if (pair.first == val.mat_name) {
                        existing_key = true;
                        val.mat = pair.second;
                        val.tp = chrono::system_clock::now();
                    }
                }

                if (!existing_key) {
                    mat_desc_row_t val;
                    val.mat = pair.second;
                    val.tp = chrono::system_clock::now();
                    val.is_displaying = {};
                    val.mat_name = pair.first;
                    matlist.at(0).append(val, true);
                }
            }
            matlist.auto_draw(true);

            // 이미지 갱신
            for (auto& proxy : matlist.at(0)) {
                auto& val = proxy.value<mat_desc_row_t>();
                if (val.is_displaying) {
                    if (val.is_displaying.value()) {
                        imshow(val.mat_name, val.mat);
                    }
                    else {
                        val.is_displaying = {};
                        cv::destroyWindow(val.mat_name);
                    }
                }
            }

            // 틱 미터 갱신
            tickmeters.auto_draw(false);
            {
                int order = 1;
                for (auto& items : tickmeters.at(0)) {
                    items.text(0, "-");
                    items.text(2, "-");
                }

                for (auto& ticks : g_recognizer.get_latest_timings()) {
                    auto tick = ticks.second.count();
                    char buf[24];
                    snprintf(buf, sizeof buf, "%d.%03d ms", tick / 1000, tick % 1000);

                    auto finder = [&]() {
                        for (auto& row : tickmeters.at(0)) {
                            if (row.text(1) == ticks.first) {
                                return optional{row};
                            }
                        }
                        return optional<listbox::item_proxy>{};
                    };

                    if (auto found = finder()) {
                        found->text(0, to_string(order++));
                        found->text(2, buf);
                    }
                    else {
                        tickmeters.at(0).append({to_string(order++), ticks.first, (string)buf});
                    }
                }
            }
            tickmeters.auto_draw(true);
        }
    });
    tm_waitkey.start();

    // -- 스냅샷 관련
    optional<billiards::recognizer_t::parameter_type> snapshot;
    timer snapshot_loader{100ms};
    button btn_snap_load(fm), btn_snapshot(fm), btn_snap_abort(fm);
    btn_snap_load.caption("Load Snapshot (Alt+E)");
    btn_snapshot.caption("Capture Snapshot (Alt+C)");
    btn_snap_abort.caption("Abort Snapshot");

    btn_snap_abort.events().click([&](auto) { snapshot.reset(); });
    btn_snap_load.events().click([&](auto) {
        filebox fb(fm, true);
        fb.add_filter("Ar Billiards Snapshot Format", "*.arbsnap");
        fb.allow_multi_select(false);

        if (auto paths = fb(); paths.empty() == false) {
            auto path = paths.front().string();
            ifstream strm{path, ios::binary | ios::in};

            if (strm.is_open()) {
                strm >> snapshot.emplace();
                snapshot_loader.start();
            }
        }
    });
    btn_snapshot.events().click([&](auto) {
        auto snap = g_recognizer.get_image_snapshot();
        filebox fb(fm, false);

        for (int i = 0; i < 100; ++i) {
            auto start_path_str = fb.path().string();
            start_path_str.pop_back();
            auto file_name = curtime_prefix + " ("s + to_string(i) + ").arbsnap"s;
            auto start_path = start_path_str / filesystem::path(file_name);
            if (!filesystem::exists(start_path)) {
                fb.init_file(file_name);
                break;
            }
        }

        fb.add_filter("Ar Billiards Snapshot Format", "*.arbsnap");
        fb.allow_multi_select(false);

        if (auto paths = fb(); paths.empty() == false) {
            auto path = paths.front().string();
            ofstream strm{path, ios::binary | ios::out};
            strm << snap;
        }
    });
    snapshot_loader.elapse([&]() {
        if (snapshot) {
            g_recognizer.refresh_image(*snapshot, [](auto&, auto&) { void ui_on_refresh(); ui_on_refresh(); });
        }
        else {
            snapshot_loader.stop();
        }
    });

    // -- 로딩
    button btn_video_load(fm), btn_video_record(fm), btn_video_playpause(fm);
    vector<video_frame_chunk> frame_chunks;
    optional<video_frame> previous;
    size_t frame_index = 0;
    bool is_playing_video = false;
    btn_video_load.caption("Load Video");
    btn_video_record.caption("Record Video");
    btn_video_playpause.caption("Play");

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
            auto& path = paths.front();
            n->video.strm_out = make_shared<ofstream>(path.string(), ios::out | ios::binary);
            n->video.pivot_time = chrono::system_clock::now();
            n->video.is_recording = true;
            btn_video_record.caption("Stop Recording");
            btn_video_record.bgcolor(colors::red);
        }
    });
    btn_video_load.events().click([&](auto) {
        if (is_playing_video) {
            msgbox("Please stop currently playing video first before load.");
            return;
        }

        filebox fb(fm, true);
        fb.add_filter("AR Billiards Video Trace Type", "*.arbtrace");
        fb.allow_multi_select(false);

        if (auto paths = fb.show(); !paths.empty()) {
            auto path = paths.front().string();
            frame_chunks.clear();
            frame_index = 0;

            ifstream in{path, ios::in | ios::binary};
            while (!in.eof()) {
                in >> frame_chunks.emplace_back();
            }
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

            video_player.interval(10ms);
            video_player.start();
        }
        else {
            btn_video_playpause.bgcolor(colors::light_gray);
            btn_video_playpause.caption("Play");
        }
    });
    video_player.elapse([&]() {
        if (frame_chunks.empty() == false) {
            if (!is_playing_video || n->video.is_busy) {
                return;
            }

            auto& vid_chnk = frame_chunks[frame_index++ % frame_chunks.size()];
            auto vid = parse(vid_chnk);
            if (previous) {
                auto tm = max(0.01f, vid.time_point - previous->time_point);
                g_recognizer.refresh_image(previous->img, [](auto&, auto&) { void ui_on_refresh(); ui_on_refresh(); });
                video_player.interval(chrono::milliseconds((int)(tm * 1000.f)));
                n->video.is_busy = true;
            }
            previous = vid;
        }
        else {
            video_player.stop();
            previous.reset();
        }
    });

    // -- 단축키 집합
    fm.register_shortkey(L's');
    fm.register_shortkey(L'o');
    fm.register_shortkey(L'e');
    fm.register_shortkey(L'q');
    fm.register_shortkey(L'c');
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
    layout.div(
      "<vert"
      "    <mat_lists>"
      "    <timings>>"
      "<vert"
      "    weight=400"
      "    <margin=[5,5,2,5] gap=5 weight=30 <btn_reset weight=15%><btn_export><btn_import>>"
      "    <margin=[0,5,5,5] gap=5 weight=30 <btn_snap_load><btn_snapshot><btn_snap_abort weight=25%>>"
      "    <margin=[0,5,5,5] gap=5 weight=30 <btn_video_load><btn_video_record><btn_video_playpause>>"
      "    <enter weight=30 margin=5>"
      "    <center margin=5>>");

    layout["center"] << tr;
    layout["enter"] << param_enter_box;
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
    layout.collocate();

    fm.events().move([&](auto) {
        auto rect = rectangle(fm.pos(), fm.size());
        g_recognizer.props["window-position"] = (array<int, 4>&)rect;
    });
    fm.events().resized([&](auto) {
        auto rect = rectangle(fm.pos(), fm.size());
        g_recognizer.props["window-position"] = (array<int, 4>&)rect;
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

void ui_on_refresh()
{
    if (auto n = n_weak.lock(); n) {
        {
            lock_guard<mutex> lock{n->shows_lock};
            n->shows.clear();
            g_recognizer.poll(n->shows);
        }

        if (n->video.is_recording) {
            assert(n->video.strm_out);
            auto time = chrono::duration<float>(chrono::system_clock::now() - n->video.pivot_time).count();
            video_frame f = {.time_point = time, .img = g_recognizer.get_image_snapshot()};
            *n->video.strm_out << f;
        }
        else if (n->video.strm_out) {
            n->video.strm_out.reset();
        }

        n->video.is_busy = false;
        n->dirty = true;
    }
}
