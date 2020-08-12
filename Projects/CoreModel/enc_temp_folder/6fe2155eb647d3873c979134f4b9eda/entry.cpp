#include <app.hpp>
#include <iostream>

using namespace std;

int main(void)
{
    tcp_server app;

    try
    {
        app.execute();

        app.open_channel(
            "0.0.0.0",
            16667,
            {},
            [](boost::system::error_code const& err, tcp_connection_desc) {
                cout << "connection established  \n";
            }, [](boost::system::error_code const& err, tcp_connection_desc, boost::asio::const_buffer buf) {
                cout.write(static_cast<const char*>(buf.data()), buf.size());
            });

        getchar();
    }
    catch (std::exception const& e)
    {
        printf(e.what());
        throw e;
    }

    return 0;
}
