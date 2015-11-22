/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * IOUtils.h
 *
 *  Created on: Nov 7, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_IO_IOUTILS_H_
#define SRC_IO_IOUTILS_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * Input and output tools
 */
class IOUtils
{
public:

	/**
	 * Clears data
	 */
	virtual ~IOUtils();

	/**
	 * Load all images from given directory
	 * OpenCV loading modes:
	 * IMREAD_UNCHANGED  = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
	 * IMREAD_GRAYSCALE  = 0,  //!< If set, always convert image to the single channel grayscale image.
	 * IMREAD_COLOR      = 1,  //!< If set, always convert image to the 3 channel BGR color image.
	 * IMREAD_ANYDEPTH   = 2,  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
	 * IMREAD_ANYCOLOR   = 4,  //!< If set, the image is read in any possible color format.
	 * IMREAD_LOAD_GDAL  = 8   //!< If set, use the gdal driver for loading the image.
	 *
	 * @param[out] vImages	loaded images
	 * @param iMode			OpenCV loading mode
	 * @param directory		file path to laod from
	 */
	static bool loadImages( std::vector<cv::Mat> &vImages, int iMode, const std::string &strDirectory );

	/**
	 * Convert given images
	 * OpenCV conversion modes:
	 * ..
	 * COLOR_BGR2GRAY
	 * (COLOR_RGB2GRAY)
	 * ..
	 *
	 * @param[in] vImages			to be converted
	 * @param[out] vConvertedImages	converted results
	 * @param iMode					OpenCV conversion mode
	 */
	static void convertImages( std::vector<cv::Mat> &vImages, std::vector<cv::Mat> &vConvertedImages, int iMode );

	/**
	 * Calculate max dimension of given images
	 *
	 * @param[in] vImages	images to calculate from
	 * @param iMaxHeight	max height
	 * @param iMaxWidth		max width
	 */
	static void getMaxImageDimensions( std::vector<cv::Mat> &vImages, int &iMaxHeight, int &iMaxWidth );

	/**
		 * Sample image to given size
		 *
		 * @param[in] images			to be sampled
		 * @param[out] iampledImages	sampled result
		 * @param iHeight				sample height
		 * @param iWidth				sample width
		 */
	static void sampleImage( cv::Mat &image, cv::Mat &sampledImage, int iHeight, int iWidth );

	/**
	 * Sample images to given size
	 *
	 * @param[in] vImages			to be sampled
	 * @param[out] vSampledImages	sampled results
	 * @param iHeight				sample height
	 * @param iWidth				sample width
	 */
	static void sampleImages( std::vector<cv::Mat> &vImages, std::vector<cv::Mat> &vSampledImages, int iHeight, int iWidth );

	/**
	 * Show given images and wait for keystroke between each.
	 */
	static void showImages( std::vector<cv::Mat> &vImages );

private:
	/**
	 * Initializes object
	 */
	IOUtils();

};

}// namespace mai

#endif /* SRC_IO_IOUTILS_H_ */
