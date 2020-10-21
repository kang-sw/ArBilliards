#pragma once
#include <opencv2/opencv.hpp>

namespace improc
{
template <typename Vec_>
void drag_contour(std::vector<Vec_>& points, Vec_ center, double direction)
{
    // 거리의 최장점을 계산
    auto max_it = std::max_element(
      points.begin(),
      points.end(),
      [&center](Vec_ const& a, Vec_ const& b) { return cv::norm(a - center, cv::NORM_L2SQR) < cv::norm(b - center, cv::NORM_L2SQR); });

    if (max_it == points.end())
        return;

    double max_dist = cv::norm(*max_it - center);
    if (max_dist != 0) {
        direction /= max_dist;

        for (auto& pt : points) {
            auto delta_vect = pt - center;
            pt += delta_vect * direction;
        }
    }
}

} // namespace improc