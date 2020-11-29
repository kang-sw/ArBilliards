#include <amp.h>

namespace {
static struct ___amp_uninitializer {
    ~___amp_uninitializer()
    {
        concurrency::amp_uninitialize();
    }
} ___uninit;
} // namespace