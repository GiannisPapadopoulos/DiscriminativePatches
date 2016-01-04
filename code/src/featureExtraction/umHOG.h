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
 * OpenCV Histogram of Oriented Gradients
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
	 * @see featureExtraction/umHOG::extractFeatures
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
			cv::Size padding = cv::Size(0,0),
			bool bApplyPCA = false);

	/**
	 * Block size and block stride have to multiples of cell size.
	 * Image size has to be multiple of block size.
	 * Bins are 1, 3, 5, 7, 9, .. with 9 as optimum.
	 * descriptorsValues.size = bins * number of cells per block * number of blocks per image
	 * Blocks per image = ((img.width / blockStride) - (block.width / blockStride - 1)) * ((img.height / blockStride) - (block.height / blockStride - 1))
	 * Descriptor values are column major.
	 *
	 * @see http://docs.opencv.org/ref/2.4/d5/d33/structcv_1_1HOGDescriptor.html
	 *
	 * @param[out] descriptorsValues	extracted features
	 * @param[in] image		from which features are extracted
	 * @param blockSize
	 * @param blockStride
	 * @param cellSize
	 * @param iNumBins
	 * @param winStride		?
	 * @param padding		?
	 * @param bApplyPCA		reduce features by applying PCA. @see featureExtraction/umPCA::decreaseHOGDescriptorCellsByPCA
	 */
	static void extractFeatures(std::vector<float> &descriptorsValues,
			cv::Mat &image,
			cv::Size blockSize = cv::Size(16,16),
			cv::Size blockStride = cv::Size(8,8),
			cv::Size cellSize = cv::Size(8,8),
			int iNumBins = 9,
			cv::Size winStride = cv::Size(0,0),
			cv::Size padding = cv::Size(0,0),
			bool bApplyPCA = false);

	/**
	 * Visualize HOG descriptor on image
	 * @see http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization#computing_the_hog_descriptor_using_opencv
	 */
	static void getHOGDescriptorVisualImage(cv::Mat &outImage,
			cv::Mat &origImg,
			std::vector<float> &descriptorValues,
			cv::Size winSize,
			cv::Size cellSize,
			cv::Size blockSize,
			cv::Size blockStride,
			int iNumBins,
			int scaleFactor,
			double vizFactor,
			bool printValue = false);

private:

	/**
	 * Initializes object
	 */
	umHOG();

};

}// namespace mai

#endif /* SRC_FEATUREEXTRACTION_UMHOG_H_ */
