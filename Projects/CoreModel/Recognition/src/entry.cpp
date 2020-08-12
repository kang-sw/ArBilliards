#include "tcp_server.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

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
                    body->json_handler(obj);

                    for (auto v : obj)
                    {
                        cout << v.begin().key();
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

int main(void)
{
    tcp_server app;

    app.initialize();

    // 이미지 파싱 요청을 수신하기 위한 채널입니다.
    {
        app.open_channel(
          "0.0.0.0",
          16667,
          tcp_server::strand_group_identical_assignment_handler(),
          [](boost::system::error_code const& err, tcp_connection_desc, tcp_server::read_handler_type& out_handler) {
              cout << "connection established for image request channel \n";
              out_handler = json_img_buffer_reader_handler({});
          },
          65536);
    }

    // TODO: 인식 프로그램의 이미지 수신 가능 상태를 리포트하는 채널입니다.
    {
        
    }

    getchar();
    app.abort();
    return 0;
}
