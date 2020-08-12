#include "tcp_server.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

using namespace std;
using nlohmann::json;

class json_img_buffer_reader_handler
{
public:
    void operator()(boost::system::error_code const& ec,
                    tcp_connection_desc connection,
                    boost::asio::const_buffer data_in)
    {
        if (ec)
        {
            cout << "connection lost.\n";
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
                    json obj = json::parse(json_raw);
                    if (body->json_handler) { body->json_handler(obj); }

                    for (auto item : obj.items())
                    {
                        char buf[1024];
                        snprintf(buf, sizeof buf, "{\"%s\" : %s}", item.key().c_str(), item.value().type_name());

                        cout << buf;
                    }

                    cout << endl;
                    json_raw.clear();
                }
            }
            else
            {
                json_raw.push_back(ch);
            }
        }
    }

    json_img_buffer_reader_handler(std::function<void(json const& parsed)>&& handler)
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
                    tcp_connection_desc connection,
                    boost::asio::const_buffer data_in)
    {
        if (ec)
        {
            cout << "connection lost\n";
            return;
        }

        size_t head = 0;
        cout << "binary file read: " << data_in.size() << '\n';
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

int main(void)
{
    tcp_server app;

    app.initialize(8);

    // 이미지 파싱 요청을 수신하기 위한 채널입니다.
    {
        app.open_channel(
          "0.0.0.0",
          16667,
          {},
          [](boost::system::error_code const& err, tcp_connection_desc, tcp_server::read_handler_type& out_handler) {
              cout << "connection established for image request channel \n";
              out_handler = json_img_buffer_reader_handler({});
          },
          65536);
    }

    // 이미지 자체를 수신하기 위한 채널입니다.
    // 4 바이트 헤더(0x00abcdef, 4 바이트 풋프린트, 4바이트 이미지 사이즈, 두 장의 이미지 버퍼가 들어있습니다.
    {
        app.open_channel(
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
    app.abort();
    return 0;
}
