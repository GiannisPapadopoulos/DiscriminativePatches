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
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

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
			std::vector<std::vector<float> > &vNegativeFeatures)
	:	m_Data(0, 0, CV_32FC1)
	,	m_Labels(0, 0, CV_32SC1)
	{
		int iFeatureSize = 0;
		if(vPositiveFeatures.size() > 0 && vNegativeFeatures.size() > 0 && vPositiveFeatures.at(0).size() != vNegativeFeatures.at(0).size())
		{
			std::cout << "[mai::UDoMLDP::setupTrainingData] ERROR! Feature size for positive and negative do not match" << std::endl;
			return;
		}

		iFeatureSize = vPositiveFeatures.at(0).size();
		std::cout << "[mai::UDoMLDP::setupTrainingData] Number of features: " << iFeatureSize << std::endl;

		std::vector<std::vector<float> > vData;
		vData.insert(std::end(vData), std::begin(vPositiveFeatures), std::end(vPositiveFeatures));
		vData.insert(std::end(vData), std::begin(vNegativeFeatures), std::end(vNegativeFeatures));

		// setup training matrices
		unsigned int iNumPatches = vData.size();

		m_Data.create(iNumPatches, iFeatureSize, CV_32FC1);
		m_Labels.create(iNumPatches, 1, CV_32SC1);
		std::cout << "[mai::UDoMLDP::setupTrainingData] Training matrix " << m_Data.rows << "x" << m_Data.cols << std::endl;
		std::cout << "[mai::UDoMLDP::setupTrainingData] Label matrix " << m_Labels.rows << "x" << m_Labels.cols << std::endl;

		for(unsigned int i = 0; i < iNumPatches ; ++i)
		{
			std::vector<float> vCurrentData = vData[i];
			for(unsigned int j = 0; j < vCurrentData.size(); ++j)
			{
				m_Data.at<float>(i, j) = vCurrentData[j];
			}

			int iLabel = Constants::SVM_NEGATIVE_LABEL;
			if(i < vPositiveFeatures.size())
			{
				iLabel = Constants::SVM_POSITIVE_LABEL;
			}
			m_Labels.at<int>(i) = iLabel;
		}
	};

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
