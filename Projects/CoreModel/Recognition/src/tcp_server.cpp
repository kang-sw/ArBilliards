#include "tcp_server.hpp"
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

    boost::container::vector<std::unique_ptr<class channel_type>> channels;
};

// ================================================================================================
class channel_type
{
public:
    channel_type(tcp_server_impl& srv, tcp::acceptor&& acpt, tcp_server::accept_strand_group_assignment_handler_type&& on_assign_strand_group, tcp_server::accept_handler_type&& on_accept) noexcept
        : srv_(srv)
        , acpt_(std::move(acpt))
        , on_assign_strand_group_(std::move(on_assign_strand_group))
        , on_accept_(std::move(on_accept))
    {
    }

    void start(size_t buflen);
    io_context& io() { return *srv_.io; }

private:
    tcp_server_impl& srv_;
    tcp::acceptor acpt_;
    tcp_server::accept_strand_group_assignment_handler_type on_assign_strand_group_;
    tcp_server::accept_handler_type on_accept_;
};

void channel_type::start(size_t buflen)
{
    struct connection_handler_type {
    public:
        struct body_type {
            channel_type& channel;
            io_context::strand* strand;
            std::shared_ptr<tcp::socket> socket;
            std::vector<char> membuf;
            tcp_server::read_handler_type on_read;

            ~body_type() noexcept = default;
        };

        std::shared_ptr<body_type> body;

    public:
        // Read handler
        void operator()(system::error_code error, size_t bytes_in)
        {
            auto& m = *body;
            tcp_connection_desc desc = {.io = m.channel.srv_.io, .socket = m.socket, .strand = m.strand};

            if (m.on_read) {
                m.on_read(error, desc, asio::buffer(m.membuf.data(), bytes_in));
            }

            if (!error) {
                if (m.strand) {
                    m.socket->async_read_some(asio::buffer(m.membuf.data(), m.membuf.size()), m.strand->wrap(*this));
                }
                else {
                    m.socket->async_read_some(asio::buffer(m.membuf.data(), m.membuf.size()), *this);
                }
            }
        }

        // Accept handler
        void operator()(system::error_code error)
        {
            auto& m = *body;
            tcp_connection_desc desc{body->channel.srv_.io, body->socket, body->strand};
            size_t strand_group_hash = 0;

            if (m.channel.on_assign_strand_group_) {
                m.channel.on_assign_strand_group_(error, desc, strand_group_hash);
            }

            // strand 그룹을 지정합니다.
            if (strand_group_hash != 0) {
                auto& srv = m.channel.srv_;
                auto found_it = srv.io_strands.find(strand_group_hash);

                if (found_it == srv.io_strands.end()) {
                    auto [it, succeeded] = srv.io_strands.try_emplace(strand_group_hash, make_unique<io_context::strand>(*srv.io));
                    found_it = it;
                }

                m.strand = found_it->second.get();
            }
            else {
                m.strand = nullptr;
            }

            if (m.channel.on_accept_) {
                desc.strand = m.strand;
                m.channel.on_accept_(error, desc, m.on_read);
            }

            if (!error && m.on_read) {
                if (m.strand) {
                    m.socket->async_read_some(asio::buffer(m.membuf.data(), m.membuf.size()), m.strand->wrap(*this));
                }
                else {
                    m.socket->async_read_some(asio::buffer(m.membuf.data(), m.membuf.size()), *this);
                }
            }

            m.channel.start(m.membuf.size());
        }

        ~connection_handler_type() noexcept = default;
    };

    connection_handler_type handler = {.body = std::make_shared<connection_handler_type::body_type>(connection_handler_type::body_type{*this, nullptr, std::make_shared<tcp::socket>(io()), std::vector<char>()})};
    handler.body->membuf.resize(buflen);
    acpt_.async_accept(*handler.body->socket, move(handler));
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

void tcp_server::open_channel(std::string_view ip_expr, uint16_t port, tcp_server::accept_strand_group_assignment_handler_type&& on_assign_strand_group, tcp_server ::accept_handler_type&& on_accept, size_t default_buffer_size)
{
    if (!pimpl_) {
        throw std::logic_error("open_channel must be called after initialization");
    }

    auto& m = *pimpl_;
    auto& channel = m.channels.emplace_back(make_unique<channel_type>(channel_type(m, tcp::acceptor(*m.io, tcp::endpoint(make_address(ip_expr), port)), std::move(on_assign_strand_group), std::move(on_accept))));

    channel->start(default_buffer_size);
}

void tcp_server::initialize(size_t num_io_thr)
{
    auto& m = *pimpl_;

    m.io = make_unique<io_context>();
    m.io_work = make_unique<io_context::work>(*m.io);

    while (num_io_thr--) {
        m.io_thr.create_thread([io = m.io.get()]() { io->run(); });
    }
}

void tcp_server::abort() noexcept
{
    auto& m = *pimpl_;
    if (m.io == nullptr) {
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

boost::asio::io_context* tcp_server::context() const
{
    return pimpl_->io.get();
}
