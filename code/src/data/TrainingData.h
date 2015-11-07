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
	TrainingData();

	/**
	 * Clears data
	 */
	virtual ~TrainingData();

	std::vector<cv::Mat> getPositives() const;
	void setPositives( std::vector<cv::Mat> vImages );

	std::vector<cv::Mat> getNegatives() const;
	void setNegatives( std::vector<cv::Mat> vImages );

	/**
	 * Create training data matrix from positive and negative images with corresponding label matrix.
	 * Images are sampled according to max dimensions if not set otherwise.
	 * @param[out] data		positive and negative images
	 * @param[out] labels	1, -1
	 */
	void getUniformTrainingData( cv::Mat &data, cv::Mat &labels );

	/**
	 * Set dimensions for sampling.
	 */
	void setUniformImageDimensions( int iHeight, int iWidth );

private:
	void setMaxImageDimensions( cv::Mat image );

	void sampleImage( cv::Mat &vimage, cv::Mat &sampledImages );

	void addImageToData( cv::Mat data, cv::Mat image, int iPos);

	std::vector<cv::Mat>	m_vPositives;
	std::vector<cv::Mat>	m_vNegatives;

	int	m_iMaxHeight;
	int	m_iMaxWidth;

};

}// namespace mai

#endif /* SRC_DATA_TRAININGDATA_H_ */
