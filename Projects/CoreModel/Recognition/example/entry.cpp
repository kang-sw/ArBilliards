// Recognition.cpp : Defines the entry point for the application.
//

#define _USE_MATH_DEFINES
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

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
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.sdk_verbose = true; // Disable verbose mode
    init_params.camera_fps = 30;

    // Open the camera
    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS)
    {
        printf("Open failed ... %d", err);
        exit(-1);
    }

    // Get camera information (ZED serial number)
    auto zed_serial = zed.getCameraInformation().serial_number;
    printf("Hello! This is my serial number: %d\n", zed_serial);

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
                cv::Scalar HLS_FILTER_MAX(110, 120, 255);

                UImage.copyTo(Image);
                cv::UMat TmpImg;
                cv::inRange(Image, HLS_FILTER_MIN, HLS_FILTER_MAX, UImage);
                cv::dilate(UImage, UImage, {}, {-1, -1}, 25);
                cv::erode(UImage, UImage, {}, {-1, -1}, 25);

                UImage.copyTo(TableMask);

                // cv::bitwise_xor(Frame, Frame, Frame, TableMask);
            }

            /* 깊이 이미지 추출 */
            if (true)
            {
                cv::Mat Depth;
                {
                    sl::Mat Captured;
                    if (auto err = zed.retrieveMeasure(Captured, sl::MEASURE::DEPTH);
                        err != sl::ERROR_CODE::SUCCESS)
                    {
                        printf("Failed to retrieve depth image by error %d\n", err);
                        continue;
                    }

                    Depth = slMat2cvMat(Captured).clone();
                    cv::resize(Depth, Depth, {Frame.cols, Frame.rows});
                }

                /* 프레임 중앙에 거리 표시 */
                {
                    int y = Depth.rows / 2, x = Depth.cols / 2;
                    float Dist = Depth.at<float>(y, x);
                    cv::circle(Frame, {x, y}, 5, {0, 0, 255}, 2);
                    cv::putText(Frame, to_string(Dist), {x, y}, cv::FONT_HERSHEY_PLAIN, 1.0, {255.f, 255.f, 255.f}, 2);
                }
            }

            /* Contour 검출 */
            vector<cv::Point> FoundContours;
            {
                // 먼저 에지 검출합니다.
                UImage = TableMask.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
                {
                    cv::erode(UImage, UImage, {});
                    cv::subtract(TableMask, UImage, UImage);
                }

                vector<vector<cv::Point>> Contours;
                vector<cv::Vec4i> Hierachy;
                cv::findContours(UImage, Contours, Hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

                Frame.copyTo(UImage);

                for (int Index = 0; Index < Contours.size(); Index++)
                {
                    auto& Contour = Contours[Index];
                    cv::approxPolyDP(vector(Contour), Contour, 10, true);

                    auto Size = cv::contourArea(Contour);
                    if (Size > 30e3 && Contour.size() == 4)
                    {
                        // 당구대를 찾았습니다.
                        cv::drawContours(UImage, Contours, Index, {255.f, 255.f, 255.f}, 2, cv::LINE_8);
                        cv::putText(UImage, "Num: "s + to_string(Contours[Index].size()) + " / Size: " + to_string(Size), Contours[Index][0], cv::FONT_HERSHEY_PLAIN, 1, {255.f, 255.f, 255.f});
                        FoundContours = move(Contour);
                        break;
                    }
                }
            }

            /* 당구대의 피봇 구하기 */
            vector<cv::Point> PivotContours;
            if (!FoundContours.empty())
            {
                // 당구대의 종횡 길이 구하기
                double constexpr DTOR = M_PI / 180.0, RTOD = 180.0 / M_PI;
                double constexpr fov_x = 90 * DTOR, fov_y = 60 * DTOR;
                double constexpr table_x = 0.96, table_y = 0.51;

                static double const span_x = atan(table_x) / fov_x, span_y = atan(table_y) / fov_y;
                int const res_x = span_x * UImage.cols, res_y = span_y * UImage.rows;

                double mult_x[] = {0.5, 1.5, 1.5, 0.5};
                double mult_y[] = {0.5, 1.5, 0.5, 1.5};
                for (int i = 0; i < 4; ++i)
                {
                    cv::Point pt = {int(mult_x[i] * res_x), int(mult_y[i] * res_y)};
                    PivotContours.push_back(pt);

                    cv::circle(UImage, pt, 5, {0, 255, 255}, 2);
                }

            }

            /* Perspective 계산 */
            if(!FoundContours.empty())
            {
                
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
#endif // true
        to_wait = max<int>(1e3 / TARGET_FPS - frame_counter.getTimeMilli(), 1);
    }

    // Close the camera
    zed.close();
    return 0;
}
