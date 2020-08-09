// Recognition.cpp : Defines the entry point for the application.
//

#define _USE_MATH_DEFINES
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <optional>

using namespace std;

cv::Mat slMat2cvMat(sl::Mat& input)
{
    using namespace sl;
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType())
    {
    case MAT_TYPE::F32_C1:
        cv_type = CV_32FC1;
        break;
    case MAT_TYPE::F32_C2:
        cv_type = CV_32FC2;
        break;
    case MAT_TYPE::F32_C3:
        cv_type = CV_32FC3;
        break;
    case MAT_TYPE::F32_C4:
        cv_type = CV_32FC4;
        break;
    case MAT_TYPE::U8_C1:
        cv_type = CV_8UC1;
        break;
    case MAT_TYPE::U8_C2:
        cv_type = CV_8UC2;
        break;
    case MAT_TYPE::U8_C3:
        cv_type = CV_8UC3;
        break;
    case MAT_TYPE::U8_C4:
        cv_type = CV_8UC4;
        break;
    default:
        break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}

// CONSTANT DEFS
#define TARGET_FPS 30

int main()
{
    // Create a ZED camera object
    sl::Camera zed;

    // Set configuration parameters
    {
        sl::InitParameters init_params;
        init_params.camera_resolution = sl::RESOLUTION::HD720;
        init_params.sdk_verbose = true; // Disable verbose mode
        init_params.camera_fps = 30;
        init_params.coordinate_units = sl::UNIT::METER;

        // Open the camera
        sl::ERROR_CODE err = zed.open(init_params);
        if (err != sl::ERROR_CODE::SUCCESS)
        {
            printf("Open failed ... %d", err);
            exit(-1);
        }
    }

    // Get camera information (ZED serial number)
    auto zed_serial = zed.getCameraInformation().serial_number;
    printf("Hello! This is my serial number: %d\n", zed_serial);

    /* 위치 트래킹 활성화 */
    {
        sl::PositionalTrackingParameters parms;

        sl::ERROR_CODE err = zed.enablePositionalTracking(parms);
        if (err != sl::ERROR_CODE::SUCCESS)
        {
            printf("Position tracking enable failed ... %d", err);
            exit(-1);
        }
    }

    /* 카메라 인트린식 획득 */
    auto CameraInfo = zed.getCameraInformation({960, 540});
    cv::Mat CameraMat, DistortionMat;
    {
        auto& p = CameraInfo.camera_configuration.calibration_parameters.left_cam;
        double M[] = {p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1};
        CameraMat = cv::Mat(3, 3, CV_64FC1, M);
        DistortionMat = cv::Mat(4, 1, CV_64FC1, p.disto);
    }

    /* 메인 루프 */
    for (int val = 0, to_wait = 1; (val = (cv::waitKey(to_wait))) != 'q';)
    {
        cv::TickMeter frame_counter;
        frame_counter.start();

        if (auto err = zed.grab(); err != sl::ERROR_CODE::SUCCESS)
        {
            printf("Capture failed for code %d\n", err);
            continue;
        }

        cv::Mat Frame;
        {
            sl::Mat Captured;
            if (auto err = zed.retrieveImage(Captured, sl::VIEW::LEFT);
                err != sl::ERROR_CODE::SUCCESS)
            {
                printf("Retrieve failed for code %d\n", err);
                continue;
            }
            Frame = slMat2cvMat(Captured).clone();
        }

#if true
        cv::TickMeter tick;

        /* 이미지의 너비가 960이 되게 리사이즈 */
        cv::resize(Frame, Frame, {}, 960.0 / Frame.cols, 960.0 / Frame.cols);
        cv::Mat Image = Frame.clone();

        /* 이미지를 GPU로 migrate */
        auto UImage = Image.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::cvtColor(UImage, UImage, cv::COLOR_RGBA2BGR);

        tick.start();

        if (false) /* HSI 모드 */
        {
            /* RGB TO HSL 색공간 변환 */
            cv::cvtColor(UImage, UImage, cv::COLOR_BGR2HLS);

            /* 색공간에서 일부 엘리먼트만 추출 */
            if (false)
            {
                cv::Scalar HLS_MULT(1, 0, 1);
                cv::Scalar HLS_ADD(0, 128, 0);

                int const COLORCODE = CV_8UC3;
                cv::add(UImage.mul(cv::UMat(UImage.rows, UImage.cols, COLORCODE, HLS_MULT)), cv::UMat(UImage.rows, UImage.cols, COLORCODE, HLS_ADD), UImage);
            }

            /* Filtering 수행 */
            {
                cv::Scalar HLS_FILTER_MIN(0, 0, 150);
                cv::Scalar HLS_FILTER_MAX(25, 180, 200);
                cv::Scalar HLS_FILTER_MIN2(170, 0, 150);
                cv::Scalar HLS_FILTER_MAX2(180, 180, 200);

                UImage.copyTo(Image);
                cv::UMat TmpImg;
                cv::inRange(Image, HLS_FILTER_MIN, HLS_FILTER_MAX, UImage);
                cv::inRange(Image, HLS_FILTER_MIN2, HLS_FILTER_MAX2, TmpImg);
                cv::add(UImage, TmpImg, UImage);
                UImage.copyTo(Image);
                // UImage = UImage.mul(255);

                cv::bitwise_and(Frame, Frame, UImage, Image);
            }

            /* 출력을 위한 색공간 변환 */
            // cv::cvtColor(UImage, UImage, cv::COLOR_HLS2RGB);
        }
        else if (true) /* YUV MODE */
        {
            /* 색공간 변환 */
            cv::cvtColor(UImage, UImage, cv::COLOR_BGR2YUV);

            /* 색공간에서 UV 추출 */
            if (false)
            {
                cv::Scalar YUV_MULT(0, 1, 1);
                cv::Scalar YUV_ADD(128, 0, 0);

                int const COLORCODE = CV_8UC3;
                cv::add(UImage.mul(cv::UMat(UImage.rows, UImage.cols, COLORCODE, YUV_MULT)), cv::UMat(UImage.rows, UImage.cols, COLORCODE, YUV_ADD), UImage);
            }

            /* 필터링 */
            cv::Mat TableMask;
            {
                cv::Scalar HLS_FILTER_MIN(0, 90, 170);
                cv::Scalar HLS_FILTER_MAX(110, 140, 255);

                UImage.copyTo(Image);
                cv::inRange(Image, HLS_FILTER_MIN, HLS_FILTER_MAX, Image);

                Image.copyTo(TableMask);
            }

            /* Contour 검출 */
            vector<cv::Vec2f> FoundContours;
            {
                // 먼저 에지 검출합니다.
                Image = TableMask.clone();
                UImage = Image.getUMat(cv::ACCESS_RW);
                {
                    cv::erode(UImage, UImage, {});
                    cv::subtract(TableMask, UImage, UImage);
                }

                vector<vector<cv::Point>> Contours;
                vector<cv::Vec4i> Hierarchy;
                cv::findContours(UImage, Contours, Hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

                Frame.copyTo(UImage);

                for (int Index = 0; Index < Contours.size(); Index++)
                {
                    auto& Contour = Contours[Index];
                    cv::approxPolyDP(vector(Contour), Contour, 10, true);

                    auto Size = cv::contourArea(Contour);
                    if (Size > 20e3 && Contour.size() == 4)
                    {
                        // 당구대를 찾았습니다.
                        cv::drawContours(UImage, Contours, Index, {255.f, 255.f, 255.f}, 2, cv::LINE_8);
                        cv::putText(UImage, "Num: "s + to_string(Contours[Index].size()) + " / Size: " + to_string(Size), Contours[Index][0], cv::FONT_HERSHEY_PLAIN, 1, {255.f, 255.f, 255.f});
                        for (auto Pt : Contour)
                        {
                            FoundContours.push_back(cv::Vec2f(Pt.x, Pt.y));
                        }
                        break;
                    }
                }
            }

            /* 당구대의 피봇 구하기 - Deprecated method ! ! ! */
            if (false)
            {
                vector<cv::Vec2f> PivotContours;
                if (!FoundContours.empty())
                {
                    // 당구대의 종횡 길이 구하기
                    double constexpr DTOR = M_PI / 180.0, RTOD = 180.0 / M_PI;
                    double constexpr fov_x = 90 * DTOR, fov_y = 60 * DTOR;
                    double constexpr table_x = 0.96, table_y = 0.51;

                    static double const span_x = atan(table_x) / fov_x, span_y = atan(table_y) / fov_y;
                    int const res_x = span_x * UImage.cols, res_y = span_y * UImage.rows;

                    double mult_x[] = {0.5, 0.5, 1.5, 1.5};
                    double mult_y[] = {1.5, 0.5, 1.5, 0.5};
                    for (int i = 0; i < 4; ++i)
                    {
                        cv::Vec2f pt(mult_x[i] * res_x, mult_y[i] * res_y);
                        PivotContours.push_back(pt);

                        cv::circle(UImage, {int(pt.val[0]), int(pt.val[1])}, 5, {0, 255, 255}, 2);
                    }
                }

                /* Perspective 계산 */
                if (!FoundContours.empty())
                {
                    auto Transform = cv::getPerspectiveTransform(PivotContours, FoundContours);
                    cout << Transform << endl;

                    Transform = cv::findHomography(PivotContours, FoundContours);
                    cout << Transform << "\n----------------------------------\n";
                }
            }

            /* 카메라 포지션 계산 */
            optional<sl::Matrix4f> CameraTransform;
            {
                using namespace sl;
                sl::Pose CamPose;

                // Retrieve and transform the pose data into a new frame located at the center of the camera
                zed.getPosition(CamPose, sl::REFERENCE_FRAME::WORLD);

                if (CamPose.valid && CamPose.pose_confidence > 90)
                {
                    CameraTransform = CamPose.pose_data;
                }
            }

            /* 당구대의 위치 계산 ... PnP 메소드 활용 */
            if (FoundContours.empty() == false)
            {
                /* 당구대의 월드 포인트  */
                // findContour로 찾아진 포인트는 반드시 반시계 방향 정렬되므로 이를 고려해 3D 원점을 설정합니다. 기본 설정은 카메라 원점에 당구대 중심을 위치시키고, 수평 가로로 놓은 상태입니다.
                vector<cv::Vec3f> Pivots;
                {
                    float HalfX = 0.96 / 2;
                    float HalfZ = 0.51 / 2;

                    // OpenCV 좌표계 참조
                    Pivots.push_back({-HalfX, 0, HalfZ});
                    Pivots.push_back({-HalfX, 0, -HalfZ});
                    Pivots.push_back({HalfX, 0, -HalfZ});
                    Pivots.push_back({HalfX, 0, HalfZ});
                }

                cv::Mat RotVec, TransVec;
                cv::solvePnP(Pivots, FoundContours, CameraMat, DistortionMat, RotVec, TransVec, false, cv::SOLVEPNP_IPPE);

                auto ImgCenter = cv::sum(FoundContours) / 4;

                // 도출된 Translate Vector 표시
                {
                    stringstream ss;
                    TransVec.convertTo(TransVec, CV_32FC1);
                    ss.precision(4);

                    auto Coord = *(cv::Vec3f*)TransVec.data;

                    // 트랜슬레이션 계산
                    if (CameraTransform)
                    {
                        sl::Vector4<float> Vect(Coord.val[0], Coord.val[1], Coord.val[2], 1.f);
                        Vect = Vect * (*CameraTransform);
                        // Coord = *(cv::Vec3f*)Vect.v;

                        cv::putText(UImage, ((stringstream&)(ss << TransVec.rows << ": " << Coord)).str(), {(int)ImgCenter.val[0], (int)ImgCenter.val[1]}, cv::FONT_HERSHEY_PLAIN, 1.3, {0, 0, 255}, 2);
                    }
                }
            }

            /* 포인트 클라우드 추출 */
            if (false)
            {
                cv::Mat PtCloud;
                {
                    sl::Mat Captured;
                    if (auto err = zed.retrieveMeasure(Captured, sl::MEASURE::XYZBGRA);
                        err != sl::ERROR_CODE::SUCCESS)
                    {
                        printf("Failed to retrieve depth image by error %d\n", err);
                        continue;
                    }

                    PtCloud = slMat2cvMat(Captured).clone();
                    cv::resize(PtCloud, PtCloud, {Frame.cols, Frame.rows});
                }

                /* 프레임 중앙에 거리 표시 */
                {
                    int y = PtCloud.rows / 2, x = PtCloud.cols / 2;
                    auto Center = PtCloud.at<cv::Vec4f>(y, x);
                    Center.val[3] = 0;
                    float Dist;
                    Dist = sqrt(cv::sum(Center.mul(Center)).val[0]);

                    cv::circle(Frame, {x, y}, 5, {0, 0, 255}, 2);
                    cv::putText(Frame, to_string(Dist), {x, y}, cv::FONT_HERSHEY_PLAIN, 1.0, {255.f, 255.f, 255.f}, 2);
                }

                /* 당구대 각 귀퉁이 좌표 표시 */
                for (auto& Point : FoundContours)
                {
                    cv::Point Pt(Point.val[0], Point.val[1] - 20);
                    auto Coord = *(cv::Vec3f*)&PtCloud.at<cv::Vec4f>(Pt.y, Pt.x);

                    // 트랜슬레이션 계산
                    if (CameraTransform)
                    {
                        sl::Vector4<float> pt;
                        memcpy(pt.v, Coord.val, 3 * sizeof(float));
                        pt.w = 1.0f;

                        pt = pt * CameraTransform.value();

                        memcpy(Coord.val, pt.v, 3 * sizeof(float));
                    }
                    else
                    {
                        break;
                    }

                    cv::putText(UImage, (stringstream() << Coord).str(), Pt, cv::FONT_HERSHEY_PLAIN, 1.4f, {255, 0, 233}, 2);
                }
            }

            /* 마스크 시각화 */
            {
                cv::bitwise_xor(Frame, Frame, Frame, TableMask);
            }
        }

        /* 출력 */
        UImage.copyTo(Image);

        tick.stop();
        frame_counter.stop();

        cv::putText(Image, "Measured: "s + to_string(tick.getTimeSec()), {0, Image.rows - 45}, 0, 1, {255.f, 0.0f, 0.0f});
        cv::putText(Image, "Total: "s + to_string(frame_counter.getTimeSec()), {0, Image.rows - 5}, 0, 1, {0.f, 0.f, 255.0f});
        cv::imshow("Vlah", Image);
        cv::imshow("Source", Frame);
        UImage.release();

#endif // true
        to_wait = max<int>(1e3 / TARGET_FPS - frame_counter.getTimeMilli(), 1);
    }

    // Close the camera
    zed.close();
    return 0;
}
