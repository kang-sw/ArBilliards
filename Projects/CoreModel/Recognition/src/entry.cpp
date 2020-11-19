#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <mutex>
#include <optional>
#include <opencv2/core.hpp>
#include "templates.hxx"

#define CVUI_IMPLEMENTATION

#include <nana/gui.hpp>

#include "tcp_server.hpp"
#include "recognition.hpp"

using namespace std;
using nlohmann::json;

billiards::recognizer_t g_recognizer;
tcp_server g_app;

void ui_on_refresh();

// ================================================================================================
struct image_desc_t {
    array<float, 3> translation;
    array<float, 4> orientation;
    array<float, 16> transform;
    int rgb_w;
    int rgb_h;
    int depth_w;
    int depth_h;

    billiards::recognizer_t::camera_param_type camera;

    weak_ptr<boost::asio::ip::tcp::socket> connection;
};

struct image_chunk_t {
    vector<char> chunk;
    string_view rgb_view;
    string_view depth_view;
};

class image_retrieve_map_t
{
public:
    using value_type = pair<optional<image_desc_t>, optional<image_chunk_t>>;
    using stamp_type = int;
    using lock_type = lock_guard<mutex>;
    using container_type = map<stamp_type, value_type>;

public:
    void put_desc(stamp_type stamp, image_desc_t const& desc_in)
    {
        lock_type lck(mtx_);

        auto it = table_.try_emplace(stamp).first;
        auto& [desc, ph__] = it->second;
        desc = desc_in;

        async_try_proc_img(stamp, it);
    }

    void put_chunk(stamp_type stamp, image_chunk_t&& chunk_in)
    {
        lock_type lck(mtx_);

        auto it = table_.try_emplace(stamp).first;
        auto& [ph__, chnk] = it->second;
        chnk = move(chunk_in);

        async_try_proc_img(stamp, it);
    }

private:
    void async_try_proc_img(stamp_type stamp, container_type::iterator it)
    {
        auto& [odesc, ochnk] = it->second;

        if (odesc && ochnk) {
            if (false) {
                char buf[1024];
                snprintf(buf, sizeof buf,
                         "--- data recv successful ---\n"
                         "  stamp: %d\n"
                         "  w, h: [%d, %d]\n"
                         "  chunk size: %llu\n",
                         stamp,
                         odesc->rgb_w,
                         odesc->rgb_h,
                         ochnk->chunk.size());
                cout << buf;
            }

            billiards::recognizer_t::parameter_type image;

            image.camera_translation = *(cv::Vec3f*)&odesc->translation;
            image.camera_orientation = *(cv::Vec4f*)&odesc->orientation;
            image.camera_transform = *(cv::Matx44f*)&odesc->transform;
            cv::Point rgb_size(odesc->rgb_w, odesc->rgb_h);
            cv::Point depth_size(odesc->depth_w, odesc->depth_h);
            image.rgba.create(rgb_size, CV_8UC4);
            memcpy(image.rgba.data, ochnk->rgb_view.data(), ochnk->rgb_view.size());
            image.depth.create(depth_size, CV_32FC1);
            memcpy(image.depth.data, ochnk->depth_view.data(), ochnk->depth_view.size());

            image.camera = odesc->camera;
            auto improc_callback = [sock = odesc->connection](billiards::recognizer_t::parameter_type const& image, json const& to_send) {
                if (auto conn = sock.lock()) {
                    // 보낼 JSON 정리
                    auto p_str = make_shared<string>();
                    *p_str = to_send.dump();
                    p_str->push_back('\n');

                    conn->async_write_some(boost::asio::const_buffer(p_str->c_str(), p_str->length()), [p_str](boost::system::error_code ec, std::size_t cnt) {
                        //cout << ec << "::" << cnt << " bytes sent\n";
                    });

                    ui_on_refresh();
                }
            };
            g_recognizer.refresh_image(image, move(improc_callback));

            table_.erase(it);
        }
    }

private:
    map<stamp_type, value_type> table_;
    mutex mtx_;
};

static image_retrieve_map_t g_retrieve_map;

// ================================================================================================
class json_handler_t
{
public:
    void operator()(boost::system::error_code const& ec,
                    tcp_connection_desc connection,
                    boost::asio::const_buffer data_in)
    {
        if (ec) {
            cout << "log: image request connection lost\n";
            return;
        }

        auto& json_raw = body->json_raw;
        string_view data((char const*)data_in.data(), data_in.size());
        for (auto ch : data) {
            if (ch < 9) {
                if (!json_raw.empty()) {
                    json obj;

                    try {
                        obj = json::parse(json_raw);
                        if (body->json_handler) { body->json_handler(connection, obj); }
                        json_raw.clear();
                    } catch (json::parse_error const& perr) {
                        cout << json_raw << '\n'
                             << perr.what() << endl;
                        json_raw.clear();
                    }
                }
            }
            else {
                json_raw.push_back(ch);
            }
        }
    }

    json_handler_t(std::function<void(tcp_connection_desc const& conn, json const& parsed)>&& handler)
        : body(make_shared<body_type>())
    {
        body->json_handler = move(handler);
    }

private:
    struct body_type {
        string json_raw;
        std::function<void(tcp_connection_desc const& conn, json const& parsed)> json_handler;
    };

    shared_ptr<body_type> body;
};

class binary_recv_channel_handler
{
public:
    void operator()(boost::system::error_code const& ec,
                    tcp_connection_desc conn,
                    boost::asio::const_buffer data_in)
    {
        if (ec) {
            cout << "log: image receive connection lost\n";
            return;
        }

        auto& bin = body->bin;
        auto head = (char const*)data_in.data();
        auto const end = head + data_in.size();

        // 한 번에 여러 chunk가 입력되는 상황에도 대비합니다.
        do {
            // 16바이트의 헤더를 수집합니다
            if (bin.size() < 16) {
                auto nread = min<ptrdiff_t>(16 - bin.size(), end - head);
                bin.insert(bin.end(), head, head + nread);
                head += nread; // advance before next usage
            }

            if (bin.size() >= 16) {
                auto const header = *reinterpret_cast<array<int32_t, 4>*>(bin.data());
                if (header[0] != 0x00abcdef) {
                    cout << "error: invalid header received\n";
                    disconnect(conn);
                    return;
                }

                int64_t ntotal = header[2] + header[3];
                int64_t nto_read = ntotal - (bin.size() - 16);
                int64_t nread = min(nto_read, end - head);

                bin.insert(bin.end(), head, head + nread);
                nto_read -= nread;
                head += nread;

                if (nto_read == 0) {
                    // 모든 바이트를 읽어들이고, 완성 이미지 입력을 시도합니다.
                    image_chunk_t chnk;
                    chnk.chunk = move(bin);
                    chnk.rgb_view = string_view(chnk.chunk.data() + 16, header[2]);
                    chnk.depth_view = string_view(chnk.chunk.data() + 16 + header[2], header[3]);

                    g_retrieve_map.put_chunk(header[1], move(chnk));
                    bin = {}; // 명시적으로 moved_from 오브젝트를 초기화합니다.
                }
            }
        } while (head != end);
    }

    void disconnect(tcp_connection_desc& conn) const
    {
        if (auto psock = conn.socket.lock()) {
            psock->shutdown(boost::asio::socket_base::shutdown_both);
            psock->close();
        }
    }

    binary_recv_channel_handler()
        : body(make_shared<body_type>())
    {
    }

private:
    struct body_type {
        vector<char> bin;
    };

    shared_ptr<body_type> body;
};

static void on_image_request(tcp_connection_desc const& conn, json const& parsed)
{
    try {
        image_desc_t desc;
        desc.orientation = parsed["Orientation"];
        desc.translation = parsed["Translation"];
        desc.transform = parsed["Transform"];
        desc.rgb_h = parsed["RgbH"];
        desc.rgb_w = parsed["RgbW"];
        desc.depth_w = parsed["DepthW"];
        desc.depth_h = parsed["DepthH"];
        desc.connection = conn.socket;

        if (auto camera = parsed.find("Camera");
            camera != parsed.end()) {
            auto& c = desc.camera;
#define PARSE__(arg) camera->at(#arg).get_to(c.arg)
            PARSE__(fx);
            PARSE__(fy);
            PARSE__(cx);
            PARSE__(cy);
            PARSE__(k1);
            PARSE__(k2);
            PARSE__(p1);
            PARSE__(p2);
        }

        g_retrieve_map.put_desc(parsed["Stamp"].get<int>(), desc);
    } catch (json::type_error const& e) {
        cout << "error: " << e.what() << endl;
    }
}

void recognition_draw_ui(cv::Mat& frame);

void exec_ui();

// ================================================================================================
#include <sl/Camera.hpp>
int main(void)
{
    size_t num_thr = thread::hardware_concurrency();
    cout << "info: initializing tcp server for " << num_thr << " threads ...\n";
    g_app.initialize(num_thr);

    // 이미지 파싱 요청을 수신하기 위한 채널입니다.
    {
        g_app.open_channel(
          "0.0.0.0",
          16667,
          {},
          [](boost::system::error_code const& err, tcp_connection_desc, tcp_server::read_handler_type& out_handler) {
              cout << "info: connection established for image request channel \n";
              out_handler = json_handler_t(&on_image_request);
          },
          65536);
    }

    // 이미지 자체를 수신하기 위한 채널입니다.
    // 4 바이트 헤더(0x00abcdef, 4 바이트 풋프린트, 4바이트 컬러 이미지 사이즈, 4바이트 깊이 이미지 사이즈, 두 장의 이미지 버퍼가 들어있습니다.
    {
        g_app.open_channel(
          "0.0.0.0",
          16668,
          {},
          [](boost::system::error_code const& err, tcp_connection_desc, tcp_server::read_handler_type& out_handler) {
              cout << "info: connection established for binary receive channel \n";
              out_handler = binary_recv_channel_handler({});
          },
          2 << 20);
    }

    cout << "info: initializing recognizer ... \n";

    try {
        g_recognizer.initialize();
        exec_ui();

        g_recognizer.destroy();
        this_thread::sleep_for(100ms);
        g_app.abort();
        this_thread::sleep_for(100ms);
    } catch (exception e) {
        cout << e.what() << endl;
    }
    return 0;
}