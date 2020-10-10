#include "recognition.hpp"
#include <cvui.h>
#include <set>
#include <fstream>
#include <chrono>
#include <nana/gui.hpp>

extern billiards::recognizer_t g_recognizer;

using namespace std;

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
