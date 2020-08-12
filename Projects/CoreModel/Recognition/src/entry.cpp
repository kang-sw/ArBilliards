#include <app.hpp>

int main(void)
{
    tcp_server app;

    try
    {
        app.execute();
    }
    catch (std::exception const& e)
    {
        printf(e.what());
        return -1;
    }

    return 0;
}
