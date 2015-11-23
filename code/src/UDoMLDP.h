/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * UDoMLDP.h
 *
 *  Created on: Oct 25, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_UDOMLDP_H_
#define SRC_UDOMLDP_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

class DataSet;

/**
 * Unsupervised discovery of mid-level discriminative patches
 */
class UDoMLDP
{
public:
	/**
	 * Initializes object
	 */
	UDoMLDP();

	/**
	 * Deletes something
	 */
	virtual ~UDoMLDP();

	/**
	 * Basic algorithm:
	 * Loads positive and negative images.
	 * Computes HOG features for each image.
	 * Trains svm ousing all patches.
	 */
	void basicDetecion( std::string &strFilePathPositives, std::string &strFilePathNegatives );

	/**
	 * Main algorithm
	 */
	static void unsupervisedDiscovery( std::string &strFilePathPositives, std::string &strFilePathNegatives );

private:

	/**
	 * @see featureExtraction/cvHOG::extractFeatures
	 */
	void computeHOGForDataSet(DataSet* data,
			cv::Size imageSize,
			cv::Size blockSize,
			cv::Size blockStride,
			cv::Size cellSize,
			cv::Size winStride,
			cv::Size padding);

	/**
	 * trains a svm with all positive and negative patches
	 */
	void trainSVM();

	DataSet*	m_pPositives;
	DataSet*	m_pNegatives;

};

}// namespace mai

#endif /* SRC_UDOMLDP_H_ */
