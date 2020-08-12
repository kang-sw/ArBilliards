#include "app.hpp"
#include <atomic>
#include <unordered_map>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <boost/container/vector.hpp>

using namespace boost;
using namespace boost::asio;
using namespace boost::asio::ip;

using std::make_unique;
using std::unique_ptr;

// ================================================================================================
class connection_impl
{
public:
    unique_ptr<io_context> io;
    boost::thread_group io_thr;
    std::map<size_t, unique_ptr<io_context::strand>> io_strands;

    unique_ptr<io_context::work> io_work;

    boost::container::vector<class channel_type> channels;
};

// ================================================================================================
class channel_type
{
public:
    channel_type(tcp::acceptor&& acpt, io_context::strand& strand_ref, tcp_server::accept_handler_type&& on_accept, tcp_server::read_handler_type&& on_read) noexcept
        : acpt_(std::move(acpt))
        , strand_(strand_ref)
        , on_accept_(std::move(on_accept))
        , on_read_(std::move(on_read))
    {
    }

    void start(size_t buflen);
    io_context& io() { return static_cast<io_context&>(acpt_.get_executor().context()); }

private:
    tcp::acceptor acpt_;
    io_context::strand& strand_;
    tcp_server::accept_handler_type on_accept_;
    tcp_server::read_handler_type on_read_;
};

void channel_type::start(size_t buflen)
{
    struct connection_handler_type
    {
    public:
        channel_type& channel;
        shared_ptr<tcp::socket> socket;
        shared_ptr<std::vector<char>> membuf;

    public:
        // Read handler
        void operator()(system::error_code error, size_t bytes_in)
        {
            tcp_connection_desc desc{*socket, channel.strand_};

            channel.on_read_(error, desc, asio::buffer(membuf->data(), bytes_in));

            if (!error)
            {
                socket->async_read_some(asio::buffer(membuf->data(), membuf->size()), channel.strand_.wrap(*this));
            }
        }

        // Accept handler
        void operator()(system::error_code error)
        {
            tcp_connection_desc desc{*socket, channel.strand_};

            if (channel.on_accept_) { channel.on_accept_(error, desc); }

            if (!error)
            {
                socket->async_read_some(asio::buffer(membuf->data(), membuf->size()), channel.strand_.wrap(*this));
            }

            channel.start(membuf->size());
        }

        ~connection_handler_type() noexcept = default;
    };

    connection_handler_type handler{*this, make_shared<tcp::socket>(io()), make_shared<std::vector<char>>()};
    handler.membuf->resize(buflen);
    acpt_.async_accept(*handler.socket, strand_.wrap(move(handler)));
}

// ================================================================================================
tcp_server::tcp_server() noexcept
    : pimpl_(make_unique<connection_impl>())
{
}

tcp_server::~tcp_server() noexcept
{
    abort();
}

void tcp_server::open_channel(std::string_view ip_expr, uint16_t port, accept_handler_type&& on_accept, read_handler_type&& on_receive, size_t default_buffer_size, size_t strand_group_hash)
{
    auto& m = *pimpl_;
    auto found_it = m.io_strands.find(strand_group_hash);

    //
    if (found_it == m.io_strands.end())
    {
        auto [it, succeeded] = m.io_strands.try_emplace(strand_group_hash, make_unique<io_context::strand>(*m.io));
        found_it = it;
    }

    auto& strand = found_it->second;
    auto& channel = m.channels.emplace_back(channel_type(tcp::acceptor(*m.io, tcp::endpoint(make_address(ip_expr), port)), *strand, std::move(on_accept), std::move(on_receive)));

    channel.start(default_buffer_size);
}

void tcp_server::execute(size_t num_io_thr)
{
    auto& m = *pimpl_;

    m.io = make_unique<io_context>();
    m.io_work = make_unique<io_context::work>(*m.io);

    while (num_io_thr--)
    {
        m.io_thr.create_thread([io = m.io.get()]() { io->run(); });
    }
}

void tcp_server::abort() noexcept
{
    auto& m = *pimpl_;
    if (m.io == nullptr)
    {
        return;
    }
    m.io_work = nullptr;
    m.io->stop();
    m.io_thr.join_all();

    m.io_strands.clear();
    m.channels.clear();

    m.io.reset();
}

bool tcp_server::is_running() const
{
    return !!pimpl_->io;
}
