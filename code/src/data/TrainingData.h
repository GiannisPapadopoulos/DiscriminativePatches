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
	 * Initializes object
	 */
	TrainingData(std::vector<std::vector<float> > &vPositiveFeatures,
			std::vector<std::vector<float> > &vNegativeFeatures);

	/**
	 * Clears data
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

//	void setNegatives(std::vector<std::vector<float> > &vFeatures);
//
//	/**
//	 * Create training data matrix from positive and negative images with corresponding label matrix.
//	 * Images are sampled according to max dimensions if not set otherwise.
//	 * @param[out] data		positive and negative images, toye of CV_32FC1
//	 * @param[out] labels	1, -1, type of CV_32SC1
//	 */
//	void getUniformTrainingData( cv::Mat &data, cv::Mat &labels );
//
//	/**
//	 * Set dimensions for sampling.
//	 * Default is max dimension of all images in positives and negatives.
//	 */
//	void setUniformImageDimensions( int iHeight, int iWidth );


private:

//	void setMaxImageDimensions( cv::Mat image );
//
//	void sampleImage( cv::Mat &vimage, cv::Mat &sampledImages );
//
//	void addImageToData( cv::Mat data, cv::Mat image, int iPos);

	cv::Mat	m_Data;
	cv::Mat	m_Labels;

};

}// namespace mai

#endif /* SRC_DATA_TRAININGDATA_H_ */
