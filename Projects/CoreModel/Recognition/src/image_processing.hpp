#pragma once
#include <opencv2/opencv.hpp>

namespace billiards::imgproc
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

template <typename Ty_>
cv::Matx<Ty_, 3, 3> rodrigues(cv::Vec<Ty_, 3> v)
{
    // cv::Matx<Ty_, 3, 3> retmat;
    // cv::Rodrigues(v, retmat);
    // return retmat;

    using mat_t = cv::Matx<Ty_, 3, 3>;

    auto O = cv::norm(v);
    auto [vx, vy, vz] = (v = v / O).val;
    auto cosO = cos(O);
    auto sinO = sin(O);

    mat_t V{0, -vz, vy, vz, 0, -vx, -vy, vx, 0};
    mat_t R = cosO * mat_t::eye() + sinO * V + (Ty_(1) - cosO) * v * v.t();

    return R;
}

template <typename Ty_>
cv::Vec<Ty_, 3> rodrigues(cv::Matx<Ty_, 3, 3> m)
{
    //cv::Vec<Ty_, 3> vec;
    //cv::Rodrigues(m, vec);
    //return vec;

    auto O = acos((cv::trace(m) - (Ty_)1) / (Ty_)2);
    auto v = (Ty_(1) / (Ty_(2) * sin(O))) * cv::Vec<Ty_, 3>(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1));

    return v * O;
}

} // namespace billiards::imgproc