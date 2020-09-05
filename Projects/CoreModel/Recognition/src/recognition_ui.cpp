#include "recognition.hpp"
#include <cvui.h>

extern billiards::recognizer_t g_recognizer;

using namespace std;
using namespace cvui;

void recognition_draw_ui(cv::Mat& frame)
{
    static struct
    {
        map<string, cv::Mat> shows;
        string active_img;
    } m;

    // sugars
    struct row__
    {
        row__() { beginRow(); }
        ~row__() { endRow(); }
    };

    struct col__
    {
        col__() { beginColumn(); }
        ~col__() { endColumn(); }
    };

#define ROW    if (row__ ROW__; true)
#define COLUMN if (col__ ROW__; true)
    g_recognizer.poll(m.shows);
    frame = cv::Scalar(49, 52, 49);

    beginColumn(frame, 0, 0);

    ROW
    {
        for (auto& pair : m.shows) {
            if (button(pair.first)) {
                m.active_img = pair.first;
            }
        }
    }

    endColumn();

    // show selected image
    if (auto& it = m.shows.find(m.active_img); it != m.shows.end()) {
        cv::imshow("image", it->second);
    }
}
