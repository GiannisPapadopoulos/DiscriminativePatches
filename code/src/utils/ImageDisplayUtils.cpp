/*
 * ImageDisplayUtils.cpp
 *
 *  Created on: Nov 26, 2015
 *      Author: giannis
 */
#include "ImageDisplayUtils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

using namespace std;

mai::ImageDisplayUtils::ImageDisplayUtils()
{}
mai::ImageDisplayUtils::~ImageDisplayUtils()
{}

void mai::ImageDisplayUtils::displayImage(const std::string &windowTitle, cv::Mat image, int delayInMs) {
  namedWindow( windowTitle, WINDOW_AUTOSIZE );
  imshow(windowTitle, image);
  // Wait for a keystroke in the window
  waitKey(delayInMs);
}


