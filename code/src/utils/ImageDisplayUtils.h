
#ifndef SRC_UTILS_IMAGEDISPLAYUTILS_H_
#define SRC_UTILS_IMAGEDISPLAYUTILS_H_


#include <opencv2/core/core.hpp>

namespace mai{

class ImageDisplayUtils {

public:

	virtual ~ImageDisplayUtils();

	/**
	 * Displays the provided image in a window with the given title
	 *
	 * @param windowTitle  The title of the window where the image will be displayed
	 * @param image The image to display
	 * @param delayInMs How long the image should be displayed in milliseconds. If unspecified or 0 is passed the image will
	 * be displayed until a key is pressed
	 */
	static void displayImage(const std::string &windowTitle, cv::Mat image, int delayInMs=0);

	/**
	 * Show given images and wait for keystroke between each.
	 */
	static void showImages(const std::vector<cv::Mat> &vImages);

	/**
	 * Show given image and wait for keystroke between each.
	 */
	static void showImage( const cv::Mat &image );
	static void showImage( const cv::Mat* const image );

private:
	  ImageDisplayUtils();


};
}

#endif /* UTILS_IMAGEDISPLAYUTILS_H_ */
