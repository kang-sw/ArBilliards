#include "tcp_server.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <mutex>
#include <optional>

using namespace std;
using nlohmann::json;

// ================================================================================================
struct image_desc_t
{
    array<float, 3> translation;
    array<float, 3> orientation;
    int image_w;
    int image_h;
};

struct image_chunk_t
{
    vector<char> chunk;
    string_view image_view;
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

        async_try_proc_img(it);
    }

    void put_chunk(stamp_type stamp, image_chunk_t&& chunk_in)
    {
        lock_type lck(mtx_);

        auto it = table_.try_emplace(stamp).first;
        auto& [ph__, chnk] = it->second;
        chnk = move(chunk_in);

        async_try_proc_img(it);
    }

private:
    void async_try_proc_img(container_type::iterator it)
    {
    }

private:
    map<stamp_type, value_type> table_;
    mutex mtx_;
};

static image_retrieve_map_t g_retrieve_map;
tcp_server g_app;

// ================================================================================================
class json_handler_t
{
public:
    void operator()(boost::system::error_code const& ec,
                    tcp_connection_desc connection,
                    boost::asio::const_buffer data_in)
    {
        if (ec)
        {
            cout << "log: image request connection lost\n";
            return;
        }

        auto& json_raw = body->json_raw;
        string_view data((char const*)data_in.data(), data_in.size());
        for (auto ch : data)
        {
            if (ch == 0)
            {
                if (!json_raw.empty())
                {
                    json obj;

                    try
                    {
                        obj = json::parse(json_raw);

                        if (body->json_handler) { body->json_handler(obj); }
                    }
                    catch (json::parse_error const& perr)
                    {
                        json_raw.clear();
                        cout << perr.what();
                    }
                }
            }
            else
            {
                json_raw.push_back(ch);
            }
        }
    }

    json_handler_t(std::function<void(json const& parsed)>&& handler)
        : body(make_shared<body_type>())
    {
        body->json_handler = move(handler);
    }

private:
    struct body_type
    {
        string json_raw;
        std::function<void(json const& parsed)> json_handler;
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
        if (ec)
        {
            cout << "log: image receive connection lost\n";
            return;
        }

        auto& bin = body->bin;
        auto head = (char const*)data_in.data();
        auto const end = head + data_in.size();

        do // 한 번에 여러 chunk가 입력되는 상황에도 대비합니다.
        {
            // 16바이트의 헤더를 수집합니다
            if (bin.size() < 16)
            {
                auto nread = min<ptrdiff_t>(16 - bin.size(), end - head);
                bin.insert(bin.end(), head, head + nread);
                head += nread; // advance before next usage
            }

            if (bin.size() >= 16)
            {
                auto const& header = *reinterpret_cast<array<int32_t, 4>*>(bin.data());
                if (header[0] != 0x00abcdef)
                {
                    cout << "error: invalid header received\n";
                    disconnect(conn);
                    return;
                }

                int64_t ntotal = header[2] + header[3];
                int64_t nto_read = ntotal - (bin.size() - 16);
                int64_t nread = max(nto_read, end - head);

                bin.insert(bin.end(), head, head + nread);
                nto_read -= nread;
                head += nread;

                if (nto_read == 0)
                {
                    // 모든 바이트를 읽어들이고, 완성 이미지 입력을 시도합니다.
                    image_chunk_t chnk;
                    chnk.chunk = move(bin);
                    chnk.image_view = string_view(chnk.chunk.data(), header[2]);
                    chnk.depth_view = string_view(chnk.chunk.data() + header[2], header[3]);

                    g_retrieve_map.put_chunk(header[1], move(chnk));
                    bin.clear();
                }
            }
        } while (head != end);
    }

    void disconnect(tcp_connection_desc& conn) const
    {
        if (auto psock = conn.socket.lock())
        {
            psock->shutdown(boost::asio::socket_base::shutdown_both);
            psock->close();
        }
    }

    binary_recv_channel_handler()
        : body(make_shared<body_type>())
    {
    }

private:
    struct body_type
    {
        vector<char> bin;
    };

    shared_ptr<body_type> body;
};

static void on_image_request(json const& parsed)
{
    try
    {
        image_desc_t desc;
        desc.orientation = parsed["Orientation"].get<array<float, 3>>();
        desc.translation = parsed["Translation"].get<array<float, 3>>();
        desc.image_h = parsed["ImageH"].get<int>();
        desc.image_w = parsed["ImageW"].get<int>();

        g_retrieve_map.put_desc(parsed["Stamp"].get<int>(), desc);
    }
    catch (json::type_error const& e)
    {
        cout << "error: " << e.what();
    }
}

// ================================================================================================
int main(void)
{
    g_app.initialize(8);

    // 이미지 파싱 요청을 수신하기 위한 채널입니다.
    {
        g_app.open_channel(
          "0.0.0.0",
          16667,
          {},
          [](boost::system::error_code const& err, tcp_connection_desc, tcp_server::read_handler_type& out_handler) {
              cout << "connection established for image request channel \n";
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
              cout << "connection established for binary receive channel \n";
              out_handler = binary_recv_channel_handler({});
          },
          2 << 20);
    }

    getchar();
    g_app.abort();
    return 0;
}
