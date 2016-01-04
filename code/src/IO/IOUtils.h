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
#include <map>

#include <opencv2/core/core.hpp>

namespace mai{

class DataSet;

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
	 * Load all images from given directory containing labeled image catalogue.
	 * Subfolders define label separation, their names define the labels themselves.
	 *
	 * Uses IO/IOUtils::loadImagesOrdered
	 *
	 * @param[out] vImages	loaded images
	 * @param iMode			OpenCV loading mode
	 * @param directory		file path to load from
	 * @param bAddFlipped	add horizontally flipped version of each image
	 * @param bEqualize		apply histogram equalization on images
	 */
	static bool loadCatalogue(std::map<std::string, DataSet*> &mImages,
				int iCVLoadMode,
				const std::string &strDirectory,
				bool bAddFlipped = true,
				bool bEqualize = true);

	/**
	 * Load all images from given directory in alphabetical order.
	 *
	 * Uses IO/IOUtils::loadAndAddImage
	 *
	 * @param[out] vImages	loaded images
	 * @param iMode			OpenCV loading mode
	 * @param directory		file path to load from
	 * @param bEqualize		apply histogram equalization on images
	 */
	static bool loadImagesOrdered(std::vector<cv::Mat*> &vImages,
			std::vector<std::string> &vImageNames,
			int iMode,
			const std::string &strDirectory,
			bool bEqualize);

	/**
	 * Uses IO/IOUtils::loadImage
	 */
	static void loadAndAddImage(std::vector<cv::Mat*> &vImages,
			std::vector<std::string> &vImageNames,
			int iMode,
			const std::string &strDirectoryItem,
			bool bEqualize);

	/**
	 * Load image
	 * OpenCV loading modes:
	 * IMREAD_UNCHANGED  = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
	 * IMREAD_GRAYSCALE  = 0,  //!< If set, always convert image to the single channel grayscale image.
	 * IMREAD_COLOR      = 1,  //!< If set, always convert image to the 3 channel BGR color image.
	 * IMREAD_ANYDEPTH   = 2,  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
	 * IMREAD_ANYCOLOR   = 4,  //!< If set, the image is read in any possible color format.
	 * IMREAD_LOAD_GDAL  = 8   //!< If set, use the gdal driver for loading the image.
	 *
	 * @param[out] image	loaded image
	 * @param iMode			OpenCV loading mode
	 * @param strFileName	file path to load from
	 * @param bEqualize		apply histogram equalization on image
	 *
	 */
	static bool loadImage(cv::Mat &image,
			int iMode,
			const std::string &strFileName,
			bool bEqualize = true);

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
	 * @param directory		file path to load from
	 */
	static bool loadImages( std::vector<cv::Mat*> &vImages,
			int iMode,
			const std::string &strDirectory,
			bool bEqualize = true );

	/**
	 * Double the images by adding flipped versions
	 *
	 * @param [out] vImages		image vector
	 * @param		iFlipmode	horizontally: > 0
	 */
	static void addFlippedImages( std::vector<cv::Mat*> &vImages,
			std::vector<std::string> &vImageNames,
			int iFlipMode );

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
	static void convertImages( std::vector<cv::Mat*> &vImages,
			std::vector<cv::Mat*> &vConvertedImages,
			int iMode );

	/**
	 * Apply histogram equalization on the input images
	 *
	 * @param[in] vImages			to be converted
	 * @param[out] vConvertedImages	converted results
	 */
	static void equalizeImages( std::vector<cv::Mat*> &vImages,
			std::vector<cv::Mat*> &vConvertedImages );

	/**
	 * Calculate max dimension of given images
	 *
	 * @param[in] vImages	images to calculate from
	 * @param iMaxHeight	max height
	 * @param iMaxWidth		max width
	 */
	static void getMaxImageDimensions( std::vector<cv::Mat> &vImages,
			int &iMaxHeight,
			int &iMaxWidth );

	/**
	 * Sample image to given size
	 *
	 * @param[in] images			to be sampled
	 * @param[out] iampledImages	sampled result
	 * @param iHeight				sample height
	 * @param iWidth				sample width
	 */
	static void sampleImage( cv::Mat &image,
			cv::Mat &sampledImage,
			int iHeight,
			int iWidth );

	/**
	 * Sample images to given size
	 *
	 * @param[in] vImages			to be sampled
	 * @param[out] vSampledImages	sampled results
	 * @param iHeight				sample height
	 * @param iWidth				sample width
	 */
	static void sampleImages( std::vector<cv::Mat> &vImages,
			std::vector<cv::Mat> &vSampledImages,
			int iHeight,
			int iWidth );

	/**
	 * Show given images and wait for keystroke between each.
	 */
	static void showImages( std::vector<cv::Mat> &vImages );

	/**
	 * Show given image and wait for keystroke between each.
	 */
	static void showImage( cv::Mat &image );
	static void showImage( const cv::Mat* image );

	/**
	 * Write images as jpgs to a folder with indexed filename strFileNameBase_Index.jpg
	 *
	 * @param[in] vImages	images to write
	 * @param strPath		existing filesystem folder without trailing slash
	 * @param strFileNameBase	base part of filename
	 */
	static void writeImages( std::vector<cv::Mat*> &vImages,
			const std::string &strPath,
			const std::string &strFileNameBase );

	/**
	 * Write gradiented image as jpgs to a folder with indexed filename strFileNameBase_Index.jpg
	 *
	 * Uses featureExtraction/umHOG::getHOGDescriptorVisualImage
	 *
	 * @param[in] data		images and descriptorvalues to write
	 * @param strPath		existing filesystem folder without trailing slash
	 * @param strFileNameBase	base part of filename
	 */
	static void writeHOGImages( DataSet* data,
			const std::string &strPath,
			const std::string &strFileNameBase,
			cv::Size imageSize,
			cv::Size cellSize,
			cv::Size blockSize,
			cv::Size blockStride,
			int iNumBins,
			int scaleFactor,
			double vizFactor,
			bool printValue = false);

	/**
	 *
	 */
	static void writeMatToCSV(const cv::Mat &data,
			const std::string &strMatName);

private:
	/**
	 * Initializes object
	 */
	IOUtils();

};

}// namespace mai

#endif /* SRC_IO_IOUTILS_H_ */
