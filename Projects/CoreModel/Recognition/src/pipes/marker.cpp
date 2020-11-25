#include "marker.hpp"
#include <amp.h>
#include <amp_math.h>

using namespace concurrency;

struct billiards::pipes::table_marker_finder::impl {
    // array<>
};

pipepp::pipe_error billiards::pipes::table_marker_finder::operator()(pipepp::execution_context& ec, input_type const& in, output_type& out)
{
    PIPEPP_REGISTER_CONTEXT(ec);
    bool const option_dirty = ec.consume_option_dirty_flag();

    if (option_dirty) {
        // 컨투어 랜덤 샘플 피봇 재생성
    }

    return {};
}

billiards::pipes::table_marker_finder::table_marker_finder()
    : impl_(std::make_unique<impl>())
{
}

billiards::pipes::table_marker_finder::~table_marker_finder() = default;
