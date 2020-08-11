#include <sockpp/stream_socket.h>
#include <stdexcept>
#include "json_recv_socket.hpp"

using namespace std;

void socket_json_wrapper::initialize()
{
    if (sock_)
    {
        is_alive = true;
        async_rw_thread = thread(&socket_json_wrapper::internal_thread_loop, this);
    }
    else
    {
        throw runtime_error("socket reference is empty!");
    }
}

void socket_json_wrapper::close()
{
    is_alive = false;
    if (async_rw_thread.joinable()) { async_rw_thread.join(); }

    if (!!*this)
    {
        sock_->close();
        sock_ = nullptr;
    }
}

bool socket_json_wrapper::send(std::string_view json_str)
{
    return *this ? sock_->write(json_str.data(), json_str.size()) > 0 : false;
}

void socket_json_wrapper::internal_thread_loop()
{
    while (is_alive)
    {
        
    }
}
