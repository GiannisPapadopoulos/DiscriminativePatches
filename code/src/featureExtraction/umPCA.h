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
 * Input and output tools
 */
class umPCA
{
public:

	/**
	 * Clears data
	 */
	virtual ~umPCA();

	/**
	 *
	 */
	static void decreaseHOGDescriptorCellsByPCA(std::vector<float> &features,
			std::vector<float> &reducedFeatures,
			int iNumBins);

	/**
	 *
	 */
	static void decreaseFeatureSpacebyPCA(cv::Mat &features,
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
