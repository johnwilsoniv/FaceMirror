#include <opencv2/opencv.hpp>
#include <iostream>
#include "LandmarkDetectorFunc.h"

int main() {
    // Load test image
    cv::Mat img = cv::imread("cpp_mtcnn_test.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    std::cout << "Image loaded: " << img.rows << "x" << img.cols << std::endl;

    // Initialize MTCNN detector
    LandmarkDetector::FaceDetectorMTCNN face_detector("../repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/mtcnn_detector");

    // Detect faces
    std::vector<cv::Rect_<float>> face_detections;
    std::vector<float> confidences;

    face_detector.DetectFaces(face_detections, img, confidences);

    std::cout << "\n=== FINAL RESULT ===" << std::endl;
    std::cout << "Detected " << face_detections.size() << " faces" << std::endl;
    for (size_t i = 0; i < face_detections.size(); ++i) {
        std::cout << "  Face " << i << ": x=" << face_detections[i].x
                  << ", y=" << face_detections[i].y
                  << ", w=" << face_detections[i].width
                  << ", h=" << face_detections[i].height
                  << ", conf=" << confidences[i] << std::endl;
    }

    return 0;
}
