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
using std::shared_ptr;
using std::unique_ptr;

// ================================================================================================
class tcp_server_impl
{
public:
    std::shared_ptr<io_context> io;
    boost::thread_group io_thr;
    std::map<size_t, unique_ptr<io_context::strand>> io_strands;

    unique_ptr<io_context::work> io_work;

    boost::container::vector<class channel_type> channels;
};

// ================================================================================================
class channel_type
{
public:
    channel_type(tcp_server_impl& srv, tcp::acceptor&& acpt, tcp_server::accept_strand_group_assignment_handler_type&& on_assign_strand_group, tcp_server::accept_handler_type&& on_accept, tcp_server::read_handler_type&& on_read) noexcept
        : srv_(srv)
        , acpt_(std::move(acpt))
        , on_assign_strand_group_(std::move(on_assign_strand_group))
        , on_accept_(std::move(on_accept))
        , on_read_(std::move(on_read))
    {
    }

    void start(size_t buflen);
    io_context& io() { return static_cast<io_context&>(acpt_.get_executor().context()); }

private:
    tcp_server_impl& srv_;
    tcp::acceptor acpt_;
    tcp_server::accept_strand_group_assignment_handler_type on_assign_strand_group_;
    tcp_server::accept_handler_type on_accept_;
    tcp_server::read_handler_type on_read_;
};

void channel_type::start(size_t buflen)
{
    struct connection_handler_type
    {
    public:
        channel_type& channel;
        io_context::strand* strand;
        std::shared_ptr<tcp::socket> socket;
        std::shared_ptr<std::vector<char>> membuf;

    public:
        // Read handler
        void operator()(system::error_code error, size_t bytes_in)
        {
            tcp_connection_desc desc{channel.srv_.io, *socket, strand};

            channel.on_read_(error, desc, asio::buffer(membuf->data(), bytes_in));

            if (!error)
            {
                socket->async_read_some(asio::buffer(membuf->data(), membuf->size()), strand->wrap(*this));
            }
        }

        // Accept handler
        void operator()(system::error_code error)
        {
            tcp_connection_desc desc{channel.srv_.io, *socket, strand};
            size_t strand_group_hash = 0;

            if (channel.on_assign_strand_group_)
            {
                channel.on_assign_strand_group_(error, desc, strand_group_hash);
            }

            // strand 그룹을 지정합니다.
            {
                auto& m = channel.srv_;
                auto found_it = m.io_strands.find(strand_group_hash);

                if (found_it == m.io_strands.end())
                {
                    auto [it, succeeded] = m.io_strands.try_emplace(strand_group_hash, make_unique<io_context::strand>(*m.io));
                    found_it = it;
                }

                strand = found_it->second.get();
            }

            if (channel.on_accept_)
            {
                desc.strand = strand;
                channel.on_accept_(error, desc);
            }

            if (!error)
            {
                socket->async_read_some(asio::buffer(membuf->data(), membuf->size()), strand->wrap(*this));
            }

            channel.start(membuf->size());
        }

        connection_handler_type() noexcept = delete;
        ~connection_handler_type() noexcept = default;
    };

    connection_handler_type handler{*this, nullptr, std::make_shared<tcp::socket>(io()), std::make_shared<std::vector<char>>()};
    handler.membuf->resize(buflen);
    acpt_.async_accept(*handler.socket, move(handler));
}

// ================================================================================================
tcp_server::tcp_server() noexcept
    : pimpl_(make_unique<tcp_server_impl>())
{
}

tcp_server::~tcp_server() noexcept
{
    abort();
}

void tcp_server::open_channel(std::string_view ip_expr, uint16_t port, tcp_server::accept_strand_group_assignment_handler_type&& on_assign_strand_group, tcp_server ::accept_handler_type&& on_accept, read_handler_type&& on_receive, size_t default_buffer_size)
{
    auto& m = *pimpl_;
    auto& channel = m.channels.emplace_back(channel_type(m, tcp::acceptor(*m.io, tcp::endpoint(make_address(ip_expr), port)), std::move(on_assign_strand_group), std::move(on_accept), std::move(on_receive)));

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
