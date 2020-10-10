#include "recognition.hpp"
#include <cvui.h>
#include <set>
#include <fstream>
#include <chrono>
#include <nana/gui.hpp>

extern billiards::recognizer_t g_recognizer;

using namespace std;

static struct
{
    cv::Size wnd_sz = {640, 960};
    unordered_map<string, cv::Mat> shows;
    set<string> active_img;
    set<string> pending_destroy;

    struct
    {
        float pos = 0;
        float speed = 0;
        int prev_mouse_y = 0;

    } scroll;
} m;

template <typename Ty>
void add_trackbar(const char* name, Ty* val, int array_size, Ty min, Ty max, int width, const char* fmt = "%.1Lf", bool has_separator = true)
{
    using namespace cvui;
    beginColumn();
    if (has_separator) {
        rect(m.wnd_sz.width, 2, 0x888888);
    }
    space();
    text(name);
    space(7);

    beginRow(-1, -1, 10);

    for (int i = 0; i < array_size; ++i) {
        trackbar(width, val + i, min, max, 1, fmt);
    }
    endRow();

    space();
    space(15);

    endColumn();
}

template <typename Ty>
void add_counter(const char* name, Ty* val, int array_size, Ty step, int width)
{
    using namespace cvui;
    beginColumn(width);
    rect(m.wnd_sz.width, 2, 0x888888);
    text(name);

    space(7);

    beginRow();

    for (int i = 0; i < array_size; ++i) {
        beginColumn(width);
        counter(val + i, step);
        endColumn();
    }
    endRow();

    endColumn();
}
#define PP_CAT(a, b)      PP_CAT_I(a, b)
#define PP_CAT_I(a, b)    PP_CAT_II(~, a##b)
#define PP_CAT_II(p, res) res

#define UNIQUE_NAME(base) PP_CAT(base, __LINE__)

void recognition_draw_ui(cv::Mat& frame)
{
    // sugars
    using namespace cvui;
    struct row__ {
        row__(int w = -1, int h = -1, int p = -1) { beginRow(w, h, p); }
        ~row__() { endRow(); }
    };

    struct col__ {
        col__(int w = -1, int h = -1, int p = -1) { beginColumn(w, h, p); }
        ~col__() { endColumn(); }
    };

#define ROW(title, ...) \
    if (row__ ROW__{__VA_ARGS__}; text(title), true)
#define COLUMN(title, ...) \
    if (col__ COL__{__VA_ARGS__}; text(title), true)

#define COLUMN_TITLE(default_open, title, ...)    \
    static bool UNIQUE_NAME(BOOL) = default_open; \
    static bool UNIQUE_NAME(BOOL_PRESS) = false;  \
    if (col__ COL__{__VA_ARGS__}; UNIQUE_NAME(BOOL_PRESS) = button(m.wnd_sz.width, 40, title, DEFAULT_FONT_SCALE * 1.5, UNIQUE_NAME(BOOL) * 0x224422 + 0x222322), UNIQUE_NAME(BOOL) = UNIQUE_NAME(BOOL_PRESS) ? !UNIQUE_NAME(BOOL) : UNIQUE_NAME(BOOL))

    // 윈도우 사이즈 변경 반영
    if (!mouse(LEFT_BUTTON, IS_DOWN) && m.wnd_sz != cv::Size{frame.cols, frame.rows}) {
        cv::resize(frame, frame, m.wnd_sz, 0, 0, cv::INTER_NEAREST);
    }

    // 렌더링 이미지 retrieve
    g_recognizer.poll(m.shows);
    frame = cv::Scalar(49, 52, 49);

    // 화면 스크롤 로직
    {
        if (mouse(MIDDLE_BUTTON, IS_DOWN)) {
            m.scroll.pos = 0;
        }

        int mouse_y = mouse().y;
        int delta = mouse_y - m.scroll.prev_mouse_y;
        if (mouse(LEFT_BUTTON, IS_DOWN)) {
            m.scroll.pos = min<float>(0, m.scroll.pos + delta);
            m.scroll.speed = 0.0f;
        }
        else if (mouse(LEFT_BUTTON, UP)) {
            m.scroll.speed = delta;
        }
        else {
            m.scroll.pos = min<float>(0, m.scroll.pos + m.scroll.speed);
            m.scroll.speed *= 0.9f;
        }
        m.scroll.prev_mouse_y = mouse_y;
    }

    // UI 컨텍스트

    beginColumn(frame, 0, m.scroll.pos, -1, -1, 5);
    {
        COLUMN_TITLE(false, "window size")
        {
            text("  width: ");
            trackbar(m.wnd_sz.width, &m.wnd_sz.width, 160, 1920, 1, "%.0Lf");
            text("  height: ");
            trackbar(m.wnd_sz.width, &m.wnd_sz.height, 480, 1440, 1, "%.0Lf");

            space(15);
        }

        COLUMN_TITLE(true, "Paramter Operations")
        ROW("")
        {
            static bool is_first_run = false;
            auto& g = g_recognizer;
            if (button(m.wnd_sz.width / 3, 60, "RESET")) {
                auto base = billiards::recognizer_t();
                g.table = base.table;
                g.ball = base.ball;
            }
            if (button(m.wnd_sz.width / 3, 60, "EXPORT")) {
                ofstream fs("config.bin");
                fs.write(reinterpret_cast<const char*>(&g.table), sizeof g.table);
                fs.write(reinterpret_cast<const char*>(&g.ball), sizeof g.ball);
                fs.write(reinterpret_cast<const char*>(&g.ball), sizeof g.FOV);
                fs.flush();
            }
            if (button(m.wnd_sz.width / 3, 60, "IMPORT") || is_first_run) {
                ifstream fs("config.bin");
                if (fs.is_open()) {
                    fs.read(reinterpret_cast<char*>(&g.table), sizeof g.table);
                    fs.read(reinterpret_cast<char*>(&g.ball), sizeof g.ball);
                    fs.read(reinterpret_cast<char*>(&g.ball), sizeof g.FOV);
                }
                is_first_run = false;
            }
        }

        auto& g = g_recognizer;
        int wnd_w = m.wnd_sz.width * 0.8;
        COLUMN_TITLE(false, "Parameter - Generic")
        {
            add_trackbar("FOV", (float*)&g.FOV, 2, 20.0f, 120.0f, wnd_w / 2);
        }

        COLUMN_TITLE(false, "Parameter - Table")
        {
            add_trackbar("LPF pos alpha: ", &g.table.LPF_alpha_pos, 1, 0.0, 1.0, wnd_w, "%.3Lf");
            add_trackbar("LPF rot alpha: ", &g.table.LPF_alpha_rot, 1, 0.0, 1.0, wnd_w, "%.3Lf");
            add_trackbar("ArUco marker detection square", &g.table.aruco_detection_rect_radius_per_meter, 1, 0.0f, 1000.0f, wnd_w);
            add_trackbar("Table HSV filter min", g.table.hsv_filter_min.val, 3, 0.0, 255.0, wnd_w / 3, "%.0Lf");
            add_trackbar("Table HSV filter max", g.table.hsv_filter_max.val, 3, 0.0, 255.0, wnd_w / 3, "%.0Lf");
            add_trackbar("Recognition max taxicap error distance", &g.table.solvePnP_max_distance_error_threshold, 1, 0, 50, wnd_w, "%.0Lf");
            add_trackbar("Table contour approxPolyDP epsilon", &g.table.polydp_approx_epsilon, 1, 0.0, 100.0, wnd_w);
        }

        COLUMN_TITLE(false, "Parameter - Ball Color")
        {
            add_trackbar("Pixel size minmax", &g.ball.pixel_count_per_meter_min, 2, 0.f, 10000.f, wnd_w / 2);
            add_trackbar("Edge canny thresholds 1, 2", g.ball.edge_canny_thresholds, 2, 0.0, 300.0, wnd_w / 2);
            add_trackbar("Ball RED", (float*)g.ball.color.red, 3, 0.0f, 255.0f, wnd_w / 3, "%.0Lf");
            add_trackbar("", (float*)(g.ball.color.red + 1), 3, 0.0f, 255.0f, wnd_w / 3, "%.0Lf", false);
            add_trackbar("Ball Orange", (float*)(g.ball.color.orange + 0), 3, 0.0f, 255.0f, wnd_w / 3, "%.0Lf");
            add_trackbar("", (float*)(g.ball.color.orange + 1), 3, 0.0f, 255.0f, wnd_w / 3, "%.0Lf", false);
            add_trackbar("Ball White", (float*)(g.ball.color.white + 0), 3, 0.0f, 255.0f, wnd_w / 3, "%.0Lf");
            add_trackbar("", (float*)(g.ball.color.white + 1), 3, 0.0f, 255.0f, wnd_w / 3, "%.0Lf", false);
        }

        COLUMN_TITLE(false, "Parameter - Ball Search")
        {
            auto& s = g.ball.search;
            add_trackbar("Outer Iteration", &s.iterations, 1, 1, 20, wnd_w, "%.0Lf");
            add_trackbar("Candidates", &s.num_candidates, 1, 8, 180, wnd_w, "%.0Lf");
            add_trackbar("Contours", &s.num_max_contours, 1, 16, 256, wnd_w, "%.0Lf");
            add_trackbar("Base", &s.weight_function_base, 1, 1.0f, 1.5f, wnd_w, "%.3Lf");
            add_trackbar("Optimize Color", &s.memoization_distance_rate, 1, 0.1f, 10.0f, wnd_w, "%.3Lf");
        }

        COLUMN_TITLE(true, "Focusing Image")
        {
            // 고정폭 버튼을 각 행에 배치

            auto it = m.shows.begin();
            auto const end = m.shows.end();

            int const ENTITY_WIDTH = 120;
            int num_btn_per_row = m.wnd_sz.width / ENTITY_WIDTH;
            int actual_entity_width = m.wnd_sz.width / num_btn_per_row;

            while (it != end) {
                ROW("")
                for (int i = 0; i < m.wnd_sz.width / ENTITY_WIDTH; ++i) {
                    if (it == end) {
                        break;
                    }

                    auto& pair = *it++;
                    auto found_it = m.active_img.find(pair.first);
                    bool const is_active_image = found_it != m.active_img.end();

                    if (button(actual_entity_width, 30, pair.first, DEFAULT_FONT_SCALE, is_active_image ? 0x22dd22 : DEFAULT_BUTTON_COLOR)) {
                        if (is_active_image) {
                            m.pending_destroy.emplace(pair.first);
                            m.active_img.erase(found_it);
                        }
                        else {
                            m.active_img.emplace(pair.first);
                        }
                    }
                }
            }
        }
    }
    endColumn();

    // show selected image
    for (auto& image_disp : m.active_img) {
        if (auto& it = m.shows.find(image_disp); it != m.shows.end()) {
            cv::imshow(image_disp, it->second);
        }
    }

    for (auto& destroy : m.pending_destroy) {
        cv::destroyWindow(destroy);
    }
    m.shows.clear();
    m.pending_destroy.clear();
}

#include <nana/gui/widgets/form.hpp>
#include <nana/gui/widgets/scroll.hpp>
#include <nana/gui/widgets/treebox.hpp>
#include <nana/gui/widgets/textbox.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/widgets/listbox.hpp>

struct n_type {
    mutex shows_lock;
    unordered_map<string, cv::Mat> shows;
    nana::form fm{nana::API::make_center(800, 600)};
    nana::listbox lb{fm};
    map<string, nlohmann::json*> param_mappings;
    atomic_bool dirty;
};

static n_type* n;

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
    for (auto& pair : to.items()) {
        auto it = from.find(pair.key());
        if (it != from.end()) {
            if (pair.value().type() == json::value_t::array || pair.value().type() == json::value_t::object) {
                json_iterative_substitute(pair.value(), *it);
            }
            else if (strcmp(pair.value().type_name(), it->type_name()) == 0) {
                pair.value() = *it;
            }
        }
    }
}

void exec_ui()
{
    using namespace nana;
    n_type nn;
    n = &nn;
    form& fm = n->fm; // 주 폼

    // 기본 컨피그 파일 로드
    if (ifstream strm("config.json"); strm.is_open()) {
        try {
            json parsed = json::parse((stringstream() << strm.rdbuf()).str());

            if (auto it = parsed.find("window-position"); it != parsed.end()) {
                array<int, 4> wndPos = *it;
                fm.move((rectangle&)wndPos);
            }

            json_iterative_substitute(g_recognizer.props, parsed);
        } catch (std::exception& e) {
            cout << "Failed to load configuration file ... " << endl;
        }
    }

    treebox tr(fm);
    textbox param_enter_box(fm);

    optional<treebox::item_proxy> selected_proxy;

    param_enter_box.multi_lines(false);

    // -- JSON 파라미터 입력 창 핸들
    param_enter_box.events().text_changed([&](arg_textbox const& tb) {
        if (selected_proxy) {
            if (auto it = n->param_mappings.find(selected_proxy->key()); it != n->param_mappings.end()) {
                auto text = tb.widget.text();
                auto prop = *it->second;
                auto original_type = prop.type_name();
                prop = json::parse(text, nullptr, false);

                if (strcmp(original_type, prop.type_name()) != 0) { // 파싱에 실패하면, 아무것도 하지 않습니다.
                    tb.widget.bgcolor(colors::light_pink);
                    cout << "error: type is " << prop.type_name() << endl;
                    return;
                }

                // 파싱에 성공했다면 즉시 이를 반영합니다.
                tb.widget.bgcolor(colors::light_green);
                *it->second = prop;

                auto new_label = selected_proxy.value().text();
                new_label = new_label.substr(0, new_label.find(' '));
                new_label += "  [" + tb.widget.text() + ']';

                selected_proxy->text(new_label);

                drawing(tr).update();
            }
        }
        else {
            tb.widget.bgcolor(colors::light_gray);
        }
    });

    // -- JSON 파라미터 트리 빌드
    struct node_iterative_constr_t {
        static void exec(treebox& tree, treebox::item_proxy root, json const& elem)
        {
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
            if (arg.item.child().empty()) {
                if (auto it = n->param_mappings.find(arg.item.key());
                    arg.operated && it != n->param_mappings.end()) {
                    selected_proxy = arg.item;
                    auto selected = *it;

                    param_enter_box.select(true);
                    param_enter_box.del();
                    param_enter_box.append(selected.second->dump(), true);
                    param_enter_box.select(true);
                    param_enter_box.focus();
                    return;
                }
            }
            selected_proxy = {};
            param_enter_box.select(true), param_enter_box.del();
        });
    };
    reload_tr();

    // -- 리셋, 익스포트, 임포트 버튼 구현
    button btn_reset(fm), btn_export(fm), btn_import(fm);
    btn_reset.caption("Reset");
    btn_export.caption("Export As...");
    btn_import.caption("Import From...");

    btn_reset.events().click([&](arg_click const& arg) {
        msgbox mb(arg.window_handle, "Parameter Reset", msgbox::yes_no);
        mb.icon(msgbox::icon_question);
        mb << "Are you sure?";

        if (mb.show() == msgbox::pick_yes) {
            g_recognizer.props = billiards::recognizer_t().props;
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

    // 이벤트 바인딩
    matlist.checkable(true);
    matlist.events().checked([&](arg_listbox const& arg) {
        auto ptr = arg.item.value_ptr<mat_desc_row_t>();

        if (ptr) {
            arg.item.value<mat_desc_row_t>().is_displaying = arg.item.checked();
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
            tickmeters.erase();
            {
                int order = 1;
                for (auto& ticks : g_recognizer.get_latest_timings()) {
                    auto tick = ticks.second.count();
                    char buf[24];
                    snprintf(buf, sizeof buf, "%d.%03d", tick / 1000, tick % 1000);
                    tickmeters.at(0).append({to_string(order++), ticks.first, buf + " ms"s});
                }
            }
            tickmeters.auto_draw(true);
        }
    });
    tm_waitkey.start();

    // -- 레이아웃 설정
    place layout(fm);
    layout.div(
      "<vert"
      "    <mat_lists>"
      "    <timings>>"
      "<vert"
      "    weight=400"
      "    <margin=5 gap=5 weight=40 <btn_reset><btn_export><btn_import>>"
      "    <enter weight=30 margin=5>"
      "    <margin=5 center>>");

    layout["center"] << tr;
    layout["enter"] << param_enter_box;
    layout["btn_reset"] << btn_reset;
    layout["btn_export"] << btn_export;
    layout["btn_import"] << btn_import;
    layout["mat_lists"] << matlist;
    layout["timings"] << tickmeters;

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

    timer autosave(300000ms);
    autosave.elapse([&]() {
        ofstream strm("config.json");
        strm << g_recognizer.props.dump(4);
    });

    fm.show();
    exec();

    // Export default configuration
    {
        ofstream strm("config.json");
        strm << g_recognizer.props.dump(4);
    }

    this_thread::sleep_for(100ms);
    cout << "info: GUI Profram Expired" << endl;
    cv::destroyAllWindows();
    cv::waitKey(1);
}

void ui_on_refresh()
{
    {
        lock_guard<mutex> lock{n->shows_lock};
        n->shows.clear();
        g_recognizer.poll(n->shows);
    }

    n->dirty = true;
}
