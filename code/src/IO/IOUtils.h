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
class umSVM;

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
	 * @param strPath	filesystem item to check
	 * @return true, if strPath is a directory, false if not or if it does not exist.
	 */
	static bool getIsDirectory(const std::string &strPath);

	/**
	 * @param strPath	filesystem item to check
	 * @return true, if strPath is a regular file, false if not or if it does not exist.
	 */
	static bool getIsFile(const std::string &strPath);

	/**
	 * Create given directory if it does not already exist.
	 *
	 * @param strPath	relative or absolute path of directory.
	 */
	static bool createDirectory(const std::string &strPath );

	/**
	 * Load all images from given directory containing labeled image cataloge.
	 * Subfolders define label separation, their names define the labels themselves.
	 *
	 * Uses IO/IOUtils::loadImagesOrdered
	 *
	 * @param[out] vImages	Loaded images.
	 * @param iMode			OpenCV loading mode.
	 * @param strDirectory	File path to load from.
	 * @param bAddFlipped	Add horizontally flipped version of each image.
	 * @param bEqualize		Apply histogram equalization on images.
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
	 * @param[out] vImages	Loaded images.
	 * @param[out] vImageNames	Filenames of the images.
	 * @param iMode			OpenCV loading mode.
	 * @param strDirectory	File path to load from.
	 * @param bEqualize		Apply histogram equalization on images.
	 */
	static bool loadImagesOrdered(std::vector<cv::Mat*> &vImages,
			std::vector<std::string> &vImageNames,
			const int iMode,
			const std::string &strDirectory,
			const bool bEqualize);

	/**
	 * Uses IO/IOUtils::loadImage
	 *
	 * @param[out] vImages	Loaded images.
	 * @param[out] vImageNames	Filenames of the images.
	 * @param iMode			OpenCV loading mode.
	 * @param strDirectoryItem	File path to load from.
	 * @param bEqualize		Apply histogram equalization on images.
	 */
	static void loadAndAddImage(std::vector<cv::Mat*> &vImages,
			std::vector<std::string> &vImageNames,
			const int iMode,
			const std::string &strDirectoryItem,
			const bool bEqualize);

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
	 * @param[out] image	Loaded image.
	 * @param iMode			OpenCV loading mode.
	 * @param strFileName	File path to load from.
	 * @param bEqualize		Apply histogram equalization on images.
	 */
	static bool loadImage(cv::Mat &image,
			const int iMode,
			const std::string &strFileName,
			const bool bEqualize = true);

	/**
	 * Write images as jpgs to a folder with indexed filename strFileNameBase_Index.jpg
	 *
	 * @param[in] vImages	Images to write.
	 * @param strPath		Filesystem folder without trailing slash. Will be created, if it does not exist.
	 * @param strFileNameBase	Base part of filename.
	 */
	static void writeImages(const std::vector<cv::Mat*> &vImages,
			const std::vector<std::string> &vImageNames,
			const std::string &strPath );

	/**
	 * Write gradiented image as jpgs to a folder with indexed filename strFileNameBase_Index.jpg
	 *
	 * Uses featureExtraction/umHOG::getHOGDescriptorVisualImage
	 *
	 * @param[in] data		Images and descriptorvalues to write
	 * @param strPath		Filesystem folder without trailing slash. Will be created, if it does not exist.
	 * @param strFileNameBase	Base part of filename
	 * @param imageSize		Images will be resized to this size.
	 * @param cellSize		HOG feature extraction cell size.
	 * @param blockSize		HOG feature extraction block size.
	 * @param blockStride	HOG feature extraction block stride.
	 * @param iNumBins		HOG feature extraction gradient directions.
	 * @param scaleFactor	Scaling factor for gradient visualization.
	 * @param visFactor		Scaling factor for the output image.
	 * @param printValue	Should the average gradient value be printed per cell ?
	 */
	static void writeHOGImages(const DataSet* const data,
			const std::string &strPath,
			const std::string &strFileNameBase,
			const cv::Size imageSize,
			const cv::Size cellSize,
			const cv::Size blockSize,
			const cv::Size blockStride,
			const int iNumBins,
			const int scaleFactor,
			const double vizFactor,
			const bool printValue = false);

	/**
	 * Write SVMs to given folder.
	 *
	 * @param mSVMs		Trained SVMs
	 * @param strPath	Filesystem folder without trailing slash. Will be created, if it does not exist.
	 */
	static void writeSVMs(const std::map<std::string, umSVM*> &mSVMs,
			const std::string &strPath);

	/**
	 * Load trained SVMs from a given directory. Filenames (without ending) will be map keys.
	 *
	 * @param[out] mSVMs	Map of trained SVMs.
	 * @param strPath		Path and name of folder to load from.
	 */
	static bool loadSVMsFromDirectory(std::map<std::string, umSVM*> &mSVMs,
			const std::string &strPath);

	/**
	 * Write OpenCV mat to csv file.
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
