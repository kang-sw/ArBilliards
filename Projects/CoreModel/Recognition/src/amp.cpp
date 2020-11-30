#include <amp.h>

namespace {
struct ___amp_uninitializer {
    ~___amp_uninitializer()
    {
        concurrency::amp_uninitialize();
    }
} ___uninit;
} // namespace