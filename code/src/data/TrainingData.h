/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * TrainingData.h
 *
 *  Created on: Nov 7, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_DATA_TRAININGDATA_H_
#define SRC_DATA_TRAININGDATA_H_

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

namespace mai{

/**
 * Training data for machine learning
 */
class TrainingData
{
public:

	/**
	 * Builds the training and label matrices from the given data vectors.
	 */
	TrainingData(const std::vector<std::vector<float> > &vPositiveFeatures,
			const std::vector<std::vector<float> > &vNegativeFeatures);

	/**
	 * Nothing to delete here.
	 */
	virtual ~TrainingData()
	{};

	cv::Mat getData() const
	{
		return m_Data;
	};

	cv::Mat getLabels() const
	{
		return m_Labels;
	};

private:

	cv::Mat	m_Data;
	cv::Mat	m_Labels;

};

}// namespace mai

#endif /* SRC_DATA_TRAININGDATA_H_ */
