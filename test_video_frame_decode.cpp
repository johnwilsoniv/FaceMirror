#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

int main() {
    // Load video and extract first frame
    std::string video_path = "Patient Data/Normal Cohort/IMG_0434.MOV";
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open video!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        std::cerr << "ERROR: Could not read video frame!" << std::endl;
        return -1;
    }

    cap.release();

    std::cout << "Video frame shape: " << frame.rows << "x" << frame.cols << "x" << frame.channels() << std::endl;
    std::cout << "Frame type: " << frame.type() << " (CV_8UC3=" << CV_8UC3 << ")" << std::endl;

    // Convert to float32 for comparison with Python
    cv::Mat frame_float;
    frame.convertTo(frame_float, CV_32F);

    // Save binary
    std::ofstream outfile("/tmp/cpp_video_frame0.bin", std::ios::binary);
    outfile.write((char*)frame_float.data, frame_float.total() * frame_float.elemSize());
    outfile.close();

    std::cout << "Saved C++ video frame to /tmp/cpp_video_frame0.bin" << std::endl;
    std::cout << "Total pixels: " << frame_float.total() * frame_float.channels() << std::endl;

    // Print sample pixels
    std::cout << "Sample pixels: ";
    float* ptr = (float*)frame_float.data;
    for (int i = 0; i < 10; i++) {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
