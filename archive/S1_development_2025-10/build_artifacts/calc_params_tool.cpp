// Standalone tool to compute PDM parameters from 2D landmarks
// Replicates FaceAnalyser's second CalcParams call

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

// OpenFace includes - just PDM, not the full model
#include "PDM.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <pdm_file> <landmarks_file>" << std::endl;
        std::cerr << "  pdm_file: Path to PDM model (e.g., In-the-wild_aligned_PDM_68.txt)" << std::endl;
        std::cerr << "  landmarks_file: Text file with 136 values (x0 y0 x1 y1 ... x67 y67)" << std::endl;
        return 1;
    }

    std::string pdm_file = argv[1];
    std::string landmarks_file = argv[2];

    // Load PDM
    LandmarkDetector::PDM pdm;
    try {
        pdm.Read(pdm_file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading PDM: " << e.what() << std::endl;
        return 1;
    }

    // Read landmarks from file
    std::ifstream infile(landmarks_file);
    if (!infile.is_open()) {
        std::cerr << "Error opening landmarks file: " << landmarks_file << std::endl;
        return 1;
    }

    std::vector<float> landmark_values;
    float val;
    while (infile >> val) {
        landmark_values.push_back(val);
    }
    infile.close();

    if (landmark_values.size() != 136) {
        std::cerr << "Error: Expected 136 landmark values, got " << landmark_values.size() << std::endl;
        return 1;
    }

    // Convert to OpenCV Mat format (68 rows, 2 columns)
    cv::Mat_<float> landmarks(68, 2);
    for (int i = 0; i < 68; i++) {
        landmarks.at<float>(i, 0) = landmark_values[i * 2];      // x
        landmarks.at<float>(i, 1) = landmark_values[i * 2 + 1];  // y
    }

    // Reshape to column vector (136 x 1) for CalcParams
    cv::Mat_<float> landmarks_vec = landmarks.reshape(1, 136);

    // Call CalcParams with zero initial rotation
    cv::Vec6f params_global;
    cv::Mat_<float> params_local;
    cv::Vec3f rotation_init(0.0f, 0.0f, 0.0f);

    pdm.CalcParams(params_global, params_local, landmarks_vec, rotation_init);

    // Output results
    std::cout << "params_global:" << std::endl;
    std::cout << "  scale: " << params_global[0] << std::endl;
    std::cout << "  rx: " << params_global[1] << " (" << (params_global[1] * 180.0 / 3.14159265359) << " deg)" << std::endl;
    std::cout << "  ry: " << params_global[2] << " (" << (params_global[2] * 180.0 / 3.14159265359) << " deg)" << std::endl;
    std::cout << "  rz: " << params_global[3] << " (" << (params_global[3] * 180.0 / 3.14159265359) << " deg)" << std::endl;
    std::cout << "  tx: " << params_global[4] << std::endl;
    std::cout << "  ty: " << params_global[5] << std::endl;

    // Also output in machine-readable format
    std::cout << "CSV_FORMAT:"
              << params_global[0] << ","
              << params_global[1] << ","
              << params_global[2] << ","
              << params_global[3] << ","
              << params_global[4] << ","
              << params_global[5] << std::endl;

    return 0;
}
