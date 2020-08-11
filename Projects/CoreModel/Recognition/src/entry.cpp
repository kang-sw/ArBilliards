#include <sockpp/socket.h>
#include <nlohmann/json.hpp>
#include <base64.h>
#include <sockpp/tcp_socket.h>
#include <sockpp/tcp_acceptor.h>
#include <sstream>
#include <atomic>
#include <thread>

using namespace std;

#define RECOGNIZE_IPC_PORT 16667

int main(void)
{
    sockpp::socket_initializer socket_initializer;
    sockpp::tcp_acceptor acceptor(RECOGNIZE_IPC_PORT);
    array<char, 1024> buffer;
    string json_raw;

    atomic_bool is_alive = true;
    thread([&is_alive]() { getchar(); is_alive= false; }).detach();

    acceptor.set_non_blocking();
    cout << "TCP server is open as port " << acceptor.address().port() << '\n';
    cout << "Pres ENTER to expire process ... " << '\n';

    // 외부 루프입니다. 단 하나의 연결만을 유지하되, 연결이 중단되면 다음 연결까지 프로세스를 일시 정지합니다.
    for (; is_alive;)
    {
        this_thread::sleep_for(33ms);
        auto socket = acceptor.accept();

        if (!socket)
        {
            continue;
        }

        cout << "----- NEW CONNECTION ESTABLISHED -----\n";
        socket.set_non_blocking(true);
        socket.read_timeout(100ms);

        while (!!socket && is_alive)
        {
            // 각 JSON 청크는 null 종료로 구별합니다.
            auto nread = socket.read(buffer.data(), buffer.size());

            if (nread < 0)
            {
                cout << "Connection lost with error code: " << nread << '\n';
                break;
            }
            if (nread == 0) { continue; }

            cout << string_view(buffer.data(), nread);
            continue;
            for (size_t i = 0; i < nread; i++)
            {
                char ch = buffer[i];

                if (ch == '\n')
                {
                    if (json_raw.empty() == false)
                    {
                        string_view view(json_raw);
                        auto json = nlohmann::json::parse(view);
                        cout << json;
                        json_raw.clear();
                    }
                }
                else
                {
                    json_raw.push_back(ch);
                }
            }
        }
        cout << "----- CONNECTION LOST -----\n";
    }

    return 0;
}