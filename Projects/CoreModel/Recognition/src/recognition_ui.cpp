#include "recognition.hpp"
#include <cvui.h>
#include <set>

extern billiards::recognizer_t g_recognizer;

using namespace std;
using namespace cvui;

static struct
{
    cv::Size wnd_sz = {640, 480};
    map<string, cv::Mat> shows;
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
void add_trackbar(const char* name, Ty* val, int array_size, Ty min, Ty max, int width, const char* fmt = "%.1Lf")
{
    beginColumn();
    rect(m.wnd_sz.width, 2, 0x888888);
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

void recognition_draw_ui(cv::Mat& frame)
{
    // sugars
    struct row__
    {
        row__(int w = -1, int h = -1, int p = -1) { beginRow(w, h, p); }
        ~row__() { endRow(); }
    };

    struct col__
    {
        col__(int w = -1, int h = -1, int p = -1) { beginColumn(w, h, p); }
        ~col__() { endColumn(); }
    };

#define ROW(title, ...) \
    if (row__ ROW__{__VA_ARGS__}; text(title), true)
#define COLUMN(title, ...) \
    if (col__ ROW__{__VA_ARGS__}; text(title), true)
#define COLUMN_TITLE(title, ...) \
    if (col__ ROW__{__VA_ARGS__}; window(m.wnd_sz.width, 10, title, DEFAULT_FONT_SCALE * 1.5), space(25), true)

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
        COLUMN_TITLE("window size")
        {
            text("  width: ");
            trackbar(m.wnd_sz.width, &m.wnd_sz.width, 160, 1920, 1, "%.0Lf");
            text("  height: ");
            trackbar(m.wnd_sz.width, &m.wnd_sz.height, 480, 1440, 1, "%.0Lf");

            space(15);
        }

        COLUMN("")
        {
            COLUMN_TITLE("Focusing Image")
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

            COLUMN_TITLE("Parameters", -1, -1)
            {
                auto& g = g_recognizer;
                int wnd_w = m.wnd_sz.width * 0.8;

                if (button(m.wnd_sz.width, 30, "PARAMETER RESET")) {
                    auto base = billiards::recognizer_t();
                    g.table = base.table;
                    g.ball = base.ball;
                }

                add_trackbar("LPF pos alpha: ", &g.table.LPF_alpha_pos, 1, 0.0, 1.0, wnd_w, "%.3Lf");
                add_trackbar("LPF rot alpha: ", &g.table.LPF_alpha_rot, 1, 0.0, 1.0, wnd_w, "%.3Lf");
                add_trackbar("ArUco marker detection square", &g.table.aruco_detection_rect_radius_per_meter, 1, 0.0f, 1000.0f, wnd_w);
                add_trackbar("Table HSV filter min", g.table.hsv_filter_min.val, 3, 0.0, 255.0, wnd_w / 3, "%.0Lf");
                add_trackbar("Table HSV filter max", g.table.hsv_filter_max.val, 3, 0.0, 255.0, wnd_w / 3, "%.0Lf");
                add_trackbar("Recognition max taxicap error distance", &g.table.solvePnP_max_distance_error_threshold, 1, 0, 50, wnd_w, "%.0Lf");
                add_trackbar("Table contour approxPolyDP epsilon", &g.table.polydp_approx_epsilon, 1, 0.0, 100.0, wnd_w);
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
    m.pending_destroy.clear();
}
