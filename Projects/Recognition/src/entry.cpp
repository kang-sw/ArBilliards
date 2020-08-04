// Recognition.cpp : Defines the entry point for the application.
//

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

int main()
{
    // Create a ZED camera object
    sl::Camera zed;

    // Set configuration parameters
    sl::InitParameters init_params;
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
    int zed_serial = zed.getCameraInformation().serial_number;
    printf("Hello! This is my serial number: %d\n", zed_serial);

    for (int val = 0; (val = (cv::waitKey(1))) != 'q';)
    {
        if (auto err = zed.grab(); err != sl::ERROR_CODE::SUCCESS)
        {
            printf("Capture failed for code %d\n", err);
            continue;
        }

        cv::Mat Image;
        {
            sl::Mat Captured;
            if (auto err = zed.retrieveImage(Captured, sl::VIEW::LEFT);
                err != sl::ERROR_CODE::SUCCESS)
            {
                printf("Retrieve failed for code %d\n", err);
                continue;
            }
            Image = slMat2cvMat(Captured).clone();
        }

#if true
        cv::TickMeter tick;

        auto UImage = Image.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::cvtColor(UImage, UImage, cv::COLOR_RGBA2BGR);

        tick.start();

        if (false) // HSI 모드
        {
            /* RGB TO HSL 색공간 변환 */
            cv::cvtColor(UImage, UImage, cv::COLOR_BGR2HLS);

            /* 색공간에서 일부 엘리먼트만 추출 */
            {
                cv::Scalar HLS_MULT(1, 0, 1);
                cv::Scalar HLS_ADD(0, 128, 0);

                int const COLORCODE = CV_8UC3;
                cv::add(UImage.mul(cv::UMat(UImage.rows, UImage.cols, COLORCODE, HLS_MULT)), cv::UMat(UImage.rows, UImage.cols, COLORCODE, HLS_ADD), UImage);
            }

            /* 출력을 위한 색공간 변환 */
            cv::cvtColor(UImage, UImage, cv::COLOR_HLS2RGB);
        }

        /* YUV 모드 */
        if (true)
        {
            /* 색공간 변환 */
            cv::cvtColor(UImage, UImage, cv::COLOR_BGR2Lab);

            /* 색공간에서 UV 추출 */
            {
                cv::Scalar YUV_MULT(0, 1, 1);
                cv::Scalar YUV_ADD(128, 0, 0);

                int const COLORCODE = CV_8UC3;
                cv::add(UImage.mul(cv::UMat(UImage.rows, UImage.cols, COLORCODE, YUV_MULT)), cv::UMat(UImage.rows, UImage.cols, COLORCODE, YUV_ADD), UImage);
            }

            /* 출력을 위한 색공간 변환 */
            cv::cvtColor(UImage, UImage, cv::COLOR_Lab2RGB);
        }

        tick.stop();

        /* 출력 */
        UImage.copyTo(Image);

        cv::putText(Image, "Measured: "s + to_string(tick.getTimeSec()), {0, Image.rows - 5}, 0, 1, {255.f, 0.0f, 0.0f});
        cv::imshow("Vlah", Image);
#endif // true
    }

    // Close the camera
    zed.close();
    return 0;
}
