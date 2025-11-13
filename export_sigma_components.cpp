/*
 * Export Sigma Components from OpenFace CCNF models
 *
 * This extracts the sigma_components matrices that are used to compute
 * the Sigma covariance transformation for CCNF response maps.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

// From OpenFace LandmarkDetectorUtils.h
void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat)
{
    // Read in the number of rows, columns and the data type
    int row, col, type;

    stream.read((char*)&row, 4);
    stream.read((char*)&col, 4);
    stream.read((char*)&type, 4);

    output_mat = cv::Mat(row, col, type);
    int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
    stream.read((char*)output_mat.data, size);

    // Move to OpenCV 4.x Mat_ if needed
    if(output_mat.type() == CV_64FC1)
    {
        cv::Mat tmp;
        output_mat.convertTo(tmp, CV_32F);
        output_mat = tmp;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.dat> <output_dir>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string output_dir = argv[2];

    std::cout << "Loading model: " << model_path << std::endl;
    std::ifstream patchesFile(model_path, std::ios_base::in | std::ios_base::binary);

    if (!patchesFile.is_open()) {
        std::cerr << "Error: Could not open model file: " << model_path << std::endl;
        return 1;
    }

    // Read patch scaling (OpenFace format: double, then int for number of views)
    double patchScaling;
    patchesFile.read((char*)&patchScaling, 8);
    std::cout << "Patch scaling: " << patchScaling << std::endl;

    // Read number of views
    int numberViews;
    patchesFile.read((char*)&numberViews, 4);
    std::cout << "Number of views: " << numberViews << std::endl;

    // Read and skip centers for each view (Vec3d = 3 doubles)
    std::cout << "Reading " << numberViews << " view centers..." << std::endl;
    for (int i = 0; i < numberViews; ++i) {
        double center[3];
        patchesFile.read((char*)&center, 3 * sizeof(double));
        std::cout << "  View " << i << " center: [" << center[0] << ", " << center[1] << ", " << center[2] << "]" << std::endl;
    }

    // Read and skip visibility matrices for each view
    std::cout << "Reading " << numberViews << " visibility matrices..." << std::endl;
    for (int i = 0; i < numberViews; ++i) {
        cv::Mat visibility;
        ReadMatBin(patchesFile, visibility);
        std::cout << "  View " << i << " visibility: " << visibility.rows << "x" << visibility.cols << std::endl;
    }

    std::cout << "File position before sigma: " << patchesFile.tellg() << std::endl;

    // ===== READ SIGMA COMPONENTS =====
    int num_win_sizes;
    patchesFile.read((char*)&num_win_sizes, 4);
    std::cout << "Number of window sizes: " << num_win_sizes << std::endl;

    std::vector<int> windows(num_win_sizes);
    std::vector<std::vector<cv::Mat_<float>>> sigma_components(num_win_sizes);

    for (int w = 0; w < num_win_sizes; ++w) {
        patchesFile.read((char*)&windows[w], 4);

        int num_sigma_comp;
        patchesFile.read((char*)&num_sigma_comp, 4);

        std::cout << "  Window size " << windows[w] << " has " << num_sigma_comp << " sigma components" << std::endl;

        sigma_components[w].resize(num_sigma_comp);

        for (int s = 0; s < num_sigma_comp; ++s) {
            cv::Mat temp;
            ReadMatBin(patchesFile, temp);
            sigma_components[w][s] = temp;

            std::cout << "    Sigma[" << s << "]: " << temp.rows << "x" << temp.cols << std::endl;
        }
    }

    patchesFile.close();

    // ===== EXPORT TO NUMPY FORMAT =====
    std::cout << "\nExporting to: " << output_dir << std::endl;

    // Save window sizes
    std::string windows_path = output_dir + "/window_sizes.txt";
    std::ofstream windows_file(windows_path);
    for (int w : windows) {
        windows_file << w << "\n";
    }
    windows_file.close();
    std::cout << "Saved: " << windows_path << std::endl;

    // Save each sigma component as binary
    for (size_t w = 0; w < sigma_components.size(); ++w) {
        for (size_t s = 0; s < sigma_components[w].size(); ++s) {
            std::string filename = output_dir + "/sigma_w" + std::to_string(windows[w]) +
                                   "_c" + std::to_string(s) + ".bin";

            std::ofstream out(filename, std::ios::binary);

            // Write dimensions
            int rows = sigma_components[w][s].rows;
            int cols = sigma_components[w][s].cols;
            out.write(reinterpret_cast<char*>(&rows), sizeof(int));
            out.write(reinterpret_cast<char*>(&cols), sizeof(int));

            // Write data
            out.write(reinterpret_cast<char*>(sigma_components[w][s].data),
                     rows * cols * sizeof(float));

            out.close();
            std::cout << "Saved: " << filename << " (" << rows << "x" << cols << ")" << std::endl;
        }
    }

    std::cout << "\nDone! Sigma components exported successfully." << std::endl;
    return 0;
}
