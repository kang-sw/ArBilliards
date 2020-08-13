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
    std::weak_ptr<boost::asio::io_context> io;
    std::weak_ptr<boost::asio::ip::tcp::socket> socket;
    boost::asio::io_context::strand* strand;

    tcp_connection_desc() noexcept = delete;

    template <typename Fn>
    auto operator()(Fn&& f) const
    {
        if (!strand) { throw std::logic_error("Strand not yet assigned!\n"); }
        return strand->wrap(std::forward<Fn>(f));
    }

    operator bool() const { return !io.expired() && !socket.expired(); }
};

class tcp_server
{
public:
    /**
     *
     */
    using accept_strand_group_assignment_handler_type = std::function<
      void(
        boost::system::error_code const& ec,
        tcp_connection_desc connection,
        size_t& out_strand_group_hash)>;

    /**
     *
     */
    using read_handler_type = std::function<
      void(
        boost::system::error_code const& ec,
        tcp_connection_desc connection,
        boost::asio::const_buffer data_in)>;

    /**
     *
     */
    using accept_handler_type = std::function<
      void(
        boost::system::error_code const& ec,
        tcp_connection_desc connection,
        read_handler_type& out_read_handler)>;

public:
    static accept_strand_group_assignment_handler_type strand_group_identical_assignment_handler()
    {
        return [](boost::system::error_code const& ec,
                  tcp_connection_desc connection,
                  size_t& out_strand_group_hash) { out_strand_group_hash = clock(); };
    }

public:
    tcp_server() noexcept;
    ~tcp_server() noexcept;
    void open_channel(std::string_view ip_expr, uint16_t port, accept_strand_group_assignment_handler_type&& on_assign_strand_group, accept_handler_type&& on_accept, size_t default_buffer_size = 1024);
    void initialize(size_t num_io_threads = 1);
    void abort() noexcept;
    bool is_running() const;
    boost::asio::io_context* context() const;

private:
    std::unique_ptr<class tcp_server_impl> pimpl_;
};