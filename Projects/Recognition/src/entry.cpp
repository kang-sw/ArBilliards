// Recognition.cpp : Defines the entry point for the application.
//

#include <opencv2/core.hpp>
#include <sl/Camera.hpp>

using namespace std;
using namespace sl;

int main()
{
    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.sdk_verbose = true; // Disable verbose mode

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS)
    {
        printf("Open failed ... %d", err);
        exit(-1);
    }

    // Get camera information (ZED serial number)
    int zed_serial = zed.getCameraInformation().serial_number;
    printf("Hello! This is my serial number: %d\n", zed_serial);

    // Close the camera
    zed.close();
    return 0;
}
