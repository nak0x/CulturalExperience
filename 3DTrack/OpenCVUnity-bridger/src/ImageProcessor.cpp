#include <opencv2/opencv.hpp>
#include "ImageProcessor.hpp"

extern "C" {

void ProcessIRImage(unsigned char* data, int width, int height) {
    cv::Mat img(height, width, CV_8UC1, data);
    cv::Mat processed;

    // Simple thresholding for demo purposes
    cv::threshold(img, processed, 100, 255, cv::THRESH_BINARY);

    // Show for debug (optional)
    cv::imshow("Processed IR Image", processed);
    cv::waitKey(1);
}

}

