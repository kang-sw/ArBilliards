#pragma once
#include <functional>
#include <memory>
#include <thread>
#include <atomic>

// 의존성을 줄이기 위한 전방 선언 목록
namespace nlohman
{
class json;
}

namespace sockpp
{
class stream_socket;
}

class socket_json_wrapper
{
public:
    using socket_type = sockpp::stream_socket;
    using socket_ptr = std::unique_ptr<socket_type>;
    using json_recv_callback_t = std::function<void(std::string&)>;

public:
    socket_json_wrapper(socket_ptr ptr)
        : sock_(std::move(ptr))
    {
    }

    ~socket_json_wrapper()
    {
        close();
    }

    socket_type* operator->() const { return sock_.get(); }
    void initialize();
    void close();
    void register_recv_callback(json_recv_callback_t&& cb) { cb_ = std::move(cb); }
    bool send(std::string_view json_str);
    operator bool() const { return sock_ && *sock_; }

private:
    void internal_thread_loop();

private:
    socket_ptr sock_;
    json_recv_callback_t cb_;
    std::atomic_bool is_alive; // sock_ 오브젝트는 is_alive가 살아있는 동안엔 immutable입니다.
    std::thread async_rw_thread;
};
