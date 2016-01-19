/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * ImageUtils.h
 *
 *  Created on: Jan 19, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_UTILS_IMAGEUTILS_H_
#define SRC_UTILS_IMAGEUTILS_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * This class provides image processing methods from OpenCV
 */
class ImageUtils {
public:
	virtual ~ImageUtils();

	/**
	 * Double the images by adding flipped versions.
	 * Image names will be duplicated with an addition of "_flipped" before the ending.
	 *
	 * @param [out] vImages		Image vector.
	 * @param [out] vImageNames	Image names vector.
	 * @param		iFlipmode	Horizontally: > 0.
	 */
	static void addFlippedImages( std::vector<cv::Mat*> &vImages,
			std::vector<std::string> &vImageNames,
			const int iFlipMode );

	/**
	 * Convert given images
	 * OpenCV conversion modes:
	 * ..
	 * COLOR_BGR2GRAY
	 * (COLOR_RGB2GRAY)
	 * ..
	 *
	 * @param[in] vImages			To be converted.
	 * @param[out] vConvertedImages	Converted results.
	 * @param iMode					OpenCV conversion mode.
	 */
	static void convertImages(const std::vector<cv::Mat*> &vImages,
			std::vector<cv::Mat*> &vConvertedImages,
			const int iMode);

	/**
	 * Apply histogram equalization on the input images
	 *
	 * @param[in] vImages			To be converted.
	 * @param[out] vConvertedImages	Converted results.
	 */
	static void equalizeImages(const std::vector<cv::Mat*> &vImages,
			std::vector<cv::Mat*> &vConvertedImages);

	/**
	 * Calculate max dimension of given images
	 *
	 * @param[in] vImages	Images to calculate from.
	 * @param iMaxHeight	Max height.
	 * @param iMaxWidth		Max width.
	 */
	static void getMaxImageDimensions(const std::vector<cv::Mat> &vImages,
			int &iMaxHeight,
			int &iMaxWidth);

	/**
	 * Sample image to given size
	 *
	 * @param[in] images			To be sampled.
	 * @param[out] iampledImages	Sampled result.
	 * @param iHeight				Sample height.
	 * @param iWidth				Sample width.
	 */
	static void sampleImage(const cv::Mat &image,
			cv::Mat &sampledImage,
			const int iHeight,
			const int iWidth);

	/**
	 * Sample images to given size
	 *
	 * @param[in] vImages			To be sampled.
	 * @param[out] vSampledImages	Sampled results.
	 * @param iHeight				Sample height.
	 * @param iWidth				Sample width.
	 */
	static void sampleImages(const std::vector<cv::Mat> &vImages,
			std::vector<cv::Mat> &vSampledImages,
			const int iHeight,
			const int iWidth);

private:
	ImageUtils();

};

}// namespace mai

#endif /* SRC_UTILS_IMAGEUTILS_H_ */
