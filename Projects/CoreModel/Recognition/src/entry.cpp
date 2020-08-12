#include "tcp_server.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;

struct 
{
    
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
          [](boost::system::error_code const& err, tcp_connection_desc) {
              cout << "connection established  \n";
          },
          [](boost::system::error_code const& err, tcp_connection_desc, boost::asio::const_buffer buf) {

              if(err)
              {
                  cout << "connection lost" << endl;
                  return;
              }

              cout.write(static_cast<const char*>(buf.data()), buf.size());
              auto str = std::string_view(static_cast<char const*>(buf.data()), buf.size());

              for (auto c : str)
              {
                  if (c == 0)
                  {
                      cout << "<<Meet EOT>>\n";
                  }
              }
          },
          65536);
    }

    // TODO: 

    getchar();
    app.abort();
    return 0;
}
