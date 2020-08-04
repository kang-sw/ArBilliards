// Recognition.cpp : Defines the entry point for the application.
//

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

using namespace std;

cv::Mat slMat2cvMat(Mat& input)
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

        cv::TickMeter tick;

        auto UImage = Image.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        // UImage.convertTo(UImage, CV_32FC4, 1.0 / 255.0);

        tick.start();
        cv::cvtColor(UImage, UImage, cv::COLOR_RGBA2BGR);
        cv::cvtColor(UImage, UImage, cv::COLOR_BGR2HLS);
        // Image = Image.mul(cv::Vec3f(1.f, 0.f, 0.f));

        {
            cv::Scalar Multiply(1, 0, 1);
            cv::Scalar Add(0, 128, 0);

            int const COLORCODE = CV_8UC3;
            cv::add(UImage.mul(cv::UMat(UImage.rows, UImage.cols, COLORCODE, Multiply)), cv::UMat(UImage.rows, UImage.cols, COLORCODE, Add), UImage);
        }

        cv::cvtColor(UImage, UImage, cv::COLOR_HLS2RGB);
        tick.stop();

        UImage.copyTo(Image);

        cv::putText(Image, "Measured: "s + to_string(tick.getTimeSec()), {0, Image.rows - 5}, 0, 1, {255.f, 0.0f, 0.0f});
        cv::imshow("Vlah", Image);
    }

    // Close the camera
    zed.close();
    return 0;
}
