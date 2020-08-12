#pragma once
#include <memory>
#include <string_view>
#include <functional>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio.hpp>

namespace boost::system
{
class error_code;
} // namespace boost::system

struct tcp_connection_desc
{
    boost::asio::ip::tcp::socket& socket;
    boost::asio::io_context::strand& strand;

    template <typename Fn>
    auto operator()(Fn&& f) const
    {
        return strand.wrap(std::forward<Fn>(f));
    }
};

class tcp_server
{
public:
    using accept_handler_type = std::function<void(boost::system::error_code const& ec, tcp_connection_desc connection)>;
    using read_handler_type = std::function<void(boost::system::error_code const& ec, tcp_connection_desc connection, boost::asio::const_buffer data_in)>;

public:
    tcp_server() noexcept;
    ~tcp_server() noexcept;
    void open_channel(std::string_view ip_expr, uint16_t port, accept_handler_type&& on_accept, read_handler_type&& on_receive, size_t default_buffer_size = 1024, size_t strand_group_hash = 0);
    void execute(size_t num_io_threads = 1);
    void abort() noexcept;

private:
    std::unique_ptr<class connection_impl> pimpl_;
};