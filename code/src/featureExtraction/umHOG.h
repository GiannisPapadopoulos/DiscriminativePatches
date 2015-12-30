/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * cvHOG.h
 *
 *  Created on: Nov 18, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_FEATUREEXTRACTION_UMHOG_H_
#define SRC_FEATUREEXTRACTION_UMHOG_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

class DataSet;

/**
 * Input and output tools
 */
class umHOG
{
public:

	/**
	 * Clears data
	 */
	virtual ~umHOG();

	/**
	 * Computes features for all images in data.
	 * Adds them as DescriptorValues to the corresponding image in the DataSet.
	 * @see class DataSet::ImageWithDescriptors
	 *
	 * Method used:
	 * @see featureExtraction/cvHOG::extractFeatures
	 *
	 * @param data			from which features are extracted
	 * @param blockSize
	 * @param blockStride
	 * @param cellSize
	 * @param iNumBins
	 * @param winStride		?
	 * @param padding		?
	 */
	static void computeHOGForDataSet(DataSet* data,
			cv::Size imageSize,
			cv::Size blockSize = cv::Size(16,16),
			cv::Size blockStride = cv::Size(8,8),
			cv::Size cellSize = cv::Size(8,8),
			int iNumBins = 9,
			cv::Size winStride = cv::Size(0,0),
			cv::Size padding = cv::Size(0,0));

	/**
	 * Cell and block size have to divide image size.
	 * Cell size has to divide block size.
	 * Block stride has to be multiple of cell size.
	 * descriptorsValues.size = ((img.width / cellsize) - 1) * ((img.height/cellsize) - 1 ) * bins (9) * number of cells per block
	 *
	 * @param[out] descriptorsValues	extracted features
	 * @param[in] image			from which features are extracted
	 * @param blockSize
	 * @param blockStride
	 * @param cellSize
	 * @param iNumBins
	 * @param winStride		?
	 * @param padding		?
	 */
	static void extractFeatures(std::vector<float> &descriptorsValues,
			cv::Mat &image,
			cv::Size blockSize = cv::Size(16,16),
			cv::Size blockStride = cv::Size(8,8),
			cv::Size cellSize = cv::Size(8,8),
			int iNumBins = 9,
			cv::Size winStride = cv::Size(0,0),
			cv::Size padding = cv::Size(0,0));

	/**
	 * Visualize HOG descriptor on image
	 * @see http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization#computing_the_hog_descriptor_using_opencv
	 */
	static void getHOGDescriptorVisualImage(cv::Mat &outImage,
			cv::Mat &origImg,
			std::vector<float> &descriptorValues,
			cv::Size winSize,
			cv::Size cellSize,
			int scaleFactor,
			double vizFactor,
			int iNumCellsPerBlock = 4,
			int iNumBins = 9);

private:

	/**
	 * Initializes object
	 */
	umHOG();

};

}// namespace mai

#endif /* SRC_FEATUREEXTRACTION_UMHOG_H_ */
