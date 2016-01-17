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
	 * @param data			From which features are extracted.
	 * @param imageSize		Images will be resized to this size.
	 * @param blockSize		HOG feature extraction block size.
	 * @param blockStride	HOG feature extraction block stride.
	 * @param cellSize		HOG feature extraction cell size.
	 * @param iNumBins		HOG feature extraction gradient directions.
	 * @param winstride		HOG feature extraction window stride.
	 * @param padding		HOG feature extraction padding.
	 * @param bApplyPCA		Apply Principal Component Analysis to feature vectors ?
	 */
	static void computeHOGForDataSet(DataSet* const data,
			const cv::Size imageSize,
			const cv::Size blockSize = cv::Size(16,16),
			const cv::Size blockStride = cv::Size(8,8),
			const cv::Size cellSize = cv::Size(8,8),
			const int iNumBins = 9,
			const cv::Size winStride = cv::Size(0,0),
			const cv::Size padding = cv::Size(0,0),
			const bool bApplyPCA = false);

	/**
	 * Image will be resized to imageSize for further processing to ensure equal size.
	 * Block size and block stride have to multiples of cell size.
	 * Image size has to be multiple of block size.
	 * Bins are 1, 3, 5, 7, 9, .. with 9 as optimum.
	 * descriptorsValues.size = bins * number of cells per block * number of blocks per image
	 * Blocks per image = ((img.width / blockStride) - (block.width / blockStride - 1)) * ((img.height / blockStride) - (block.height / blockStride - 1))
	 * Descriptor values are column major.
	 *
	 * @see http://docs.opencv.org/ref/2.4/d5/d33/structcv_1_1HOGDescriptor.html
	 *
	 * @param[out] descriptorsValues	Extracted features
	 * @param[in] image		From which features are extracted
	 * @param imageSize		Images will be resized to this size.
	 * @param blockSize		HOG feature extraction block size.
	 * @param blockStride	HOG feature extraction block stride.
	 * @param cellSize		HOG feature extraction cell size.
	 * @param iNumBins		HOG feature extraction gradient directions.
	 * @param winstride		HOG feature extraction window stride.
	 * @param padding		HOG feature extraction padding.
	 * @param bApplyPCA		Reduce features by applying PCA. @see featureExtraction/umPCA::decreaseHOGDescriptorCellsByPCA
	 */
	static void extractFeatures(std::vector<float> &descriptorsValues,
			const cv::Mat &image,
			const cv::Size imageSize,
			const cv::Size blockSize = cv::Size(16,16),
			const cv::Size blockStride = cv::Size(8,8),
			const cv::Size cellSize = cv::Size(8,8),
			const int iNumBins = 9,
			const cv::Size winStride = cv::Size(0,0),
			const cv::Size padding = cv::Size(0,0),
			const bool bApplyPCA = false);

	/**
	 * Visualize HOG descriptor on image
	 * @see http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization#computing_the_hog_descriptor_using_opencv
	 *
	 * @param[out] outImage	Image containing gradient visualization.
	 * @param[in] origImage	Original image.
	 * @param[in] descriptorValues	Extracted features of the original image.
	 * @param winSize		Images will be resized to this size.
	 * @param cellSize		HOG feature extraction cell size.
	 * @param blockSize		HOG feature extraction block size.
	 * @param blockStride	HOG feature extraction block stride.
	 * @param iNumBins		HOG feature extraction gradient directions.
	 * @param scaleFactor	Scaling factor for gradient visualization.
	 * @param visFactor		Scaling factor for the output image.
	 * @param printValue	Should the average gradient value be printed per cell ?
	 */
	static void getHOGDescriptorVisualImage(cv::Mat &outImage,
			const cv::Mat &origImg,
			const std::vector<float> &descriptorValues,
			const cv::Size winSize,
			const cv::Size cellSize,
			const cv::Size blockSize,
			const cv::Size blockStride,
			const int iNumBins,
			const int scaleFactor,
			const double vizFactor,
			const bool printValue = false);

private:

	/**
	 * Initializes object
	 */
	umHOG();

};

}// namespace mai

#endif /* SRC_FEATUREEXTRACTION_UMHOG_H_ */
