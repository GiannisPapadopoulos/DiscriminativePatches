/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * umPCA.h
 *
 *  Created on: Dec 4, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_FEATUREEXTRACTION_UMPCA_H_
#define SRC_FEATUREEXTRACTION_UMPCA_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * OpenCV PCA feature reduction for HOG feature vectors
 */
class umPCA
{
public:

	/**
	 * Clears data
	 */
	virtual ~umPCA();

	/**
	 * Reduce HOG features by applying Principal Component Analysis on each image cell
	 *
	 * @param features	input feature vector
	 * @param reducedFeatures	output feature vector
	 * @param iNumBins	number of gradient directions used in HOG computation. Size of reduced feature vector.
	 */
	static void decreaseHOGDescriptorCellsByPCA(const std::vector<float> &features,
			std::vector<float> &reducedFeatures,
			const int iNumBins);

	/**
	 * @see http://docs.opencv.org/ref/2.4/d3/d8d/classcv_1_1PCA.html
	 */
	static void decreaseFeatureSpacebyPCA(const cv::Mat &features,
			cv::Mat &eigenValues,
			cv::Mat &eigenVectors,
			cv::Mat &mean);

private:

	/**
	 * Initializes object
	 */
	umPCA();

};

}// namespace mai

#endif /* SRC_FEATUREEXTRACTION_UMPCA_H_ */
