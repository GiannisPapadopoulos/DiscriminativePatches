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
class umSVM;

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
	void basicDetecion(std::string &strFilePathPositives, std::string &strFilePathNegatives);

	/**
	 * Main algorithm
	 */
	static void unsupervisedDiscovery(std::string &strFilePathPositives, std::string &strFilePathNegatives);

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
	 * Predict dataset
	 * @see svm/umSVM::predict
	 */
	void predictDataSetbySVM(DataSet* data);

	/**
	 * !! Check the implementation on which loading method to use !!
	 * @see svm/umSVM::train
	 */
	void trainSVMOnDataSets(DataSet* positives,
			DataSet* negatives);

	/**
	 * Construct Mat for trainingdata.
	 * Calls collectTrainingDataAndLabels for positives and negatives
	 */
	void setupTrainingData(DataSet* positives,
			DataSet* negatives,
			cv::Mat &trainingData,
			cv::Mat &labels);
	/**
	 * Put the descriptorValues from the DataSet into TrainingData and setup lables like given
	 * @param[in] data				positive or negative training data
	 * @param[out] vTrainingData	training data matrix
	 * @param[out] vLabel			labels matrix
	 * @param fLabel				labels for this set, e.g. 1.0 if positive or 0.0 if negative
	 */
	void collectTrainingDataAndLabels(DataSet* data,
			std::vector<float> &vTrainingData,
			std::vector<float> &vLabels,
			float fLabel);

	void setupTrainingDataForSinglePatchImage(DataSet* positives,
				DataSet* negatives,
				cv::Mat &trainingData,
				cv::Mat &labels);

	int collectTrainingDataAndLabelsForSingelPatchImage(DataSet* data,
			std::vector<std::vector<float> > &vTrainingData,
			std::vector<float> &vLabels,
			float fLabel);

	DataSet*	m_pPositiveTrain;
	DataSet*	m_pNegativeTrain;

	DataSet*	m_pPositiveValid;
	DataSet*	m_pNegativeValid;

	umSVM*		m_pSVM;

};

}// namespace mai

#endif /* SRC_UDOMLDP_H_ */
