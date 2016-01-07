/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * CatalogueDetection.h
 *
 *  Created on: Dec 28, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_CATALOGUEDETECTION_H_
#define SRC_CATALOGUEDETECTION_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

class Configuration;
class DataSet;
class TrainingData;
class umSVM;

/**
 * Train classifiers and predict sampled images on a catalogue of labeled images.
 *
 */
class CatalogueDetection
{
public:

	/**
	 * Load image catalogue according to configuration
	 */
	CatalogueDetection(Configuration* config);

	/**
	 * Deletes everything
	 */
	virtual ~CatalogueDetection();

	/**
	 * Processing pipeline:
	 * - Refine images though histogram equalization
	 * - Compute HOG features
	 * - Setup training and validation data
	 * - Train SVMs
	 * - Predict data
	 */
	void processPipeline();

private:

	/**
	 * @see utils/FaceDetection::detectFaces
	 */
	void detectFaces();

	/**
	 * Computes HOG descriptors for all Datasets in catalogue.
	 * @see featureExtraction/umHOG::computeHOGForDataSet
	 *
	 * param bWriteHOGImages	write HOG visualization to local folder "out"
	 */
	void computeHOG(cv::Size imageSize,
				cv::Size blockSize,
				cv::Size blockStride,
				cv::Size cellSize,
				int iNumBins,
				cv::Size winStride,
				cv::Size padding,
				bool bWriteHOGImages = false,
				bool bApplyPCA = false);

	/**
	 * Setup training data and train svms.
	 * @see svm/umSVM::train
	 *
	 * Saves trained svms by catalogue category labels.
	 *
	 * @param iDataSetDivider	divider of dataset size defining validation part, e.g. 4 -> 1/4 of patches will be in validation set.
	 * @param bSearchSupportVectors verify support vector existance in training data.
	 * @return	training was successful.
	 */
	bool trainSVMs(int iDataSetDivider = 1,
			bool bSearchSupportVectors = false);

	/**
	 * Assigns part of the descriptor vectors of each dataset for validation purpose, the rest for training.
	 *
	 * @param iDataSetDivider	divider of dataset size defining validation part, e.g. 4 -> 1/4 of patches will be in validation set.
	 */
	void divideDataSets(std::map<std::string, std::vector<std::vector<float> > > &mTrain,
			std::map<std::string, std::vector<std::vector<float> > > &mValidate,
			int iDataSetDivider);

	/**
	 * Collects random samples for each positive descriptor set from all other descriptor sets at equal part.
	 * Size of the sampled descriptor sets are the same as the positive one to have an equal amount of positive and negative training samples.
	 * (Almost the same at least, as division rest is not taken into account.)
	 *
	 * @TODO what if other set is too small ?
	 */
	void collectRandomNegatives(std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives);

	/**
	 * Creates training data from positive and negative samples.
	 * @see data/TrainingData
	 */
	void setupTrainingData(std::map<std::string, TrainingData*> &mTrain,
			std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives);

	/**
	 * Predicts data divided for each categorized svm.
	 */
	void predict(std::map<std::string, TrainingData*> &mData,
			std::map<std::string, cv::Mat> &mResults);


	/**
	 * Original images and extraacted feature vectors per named category
	 */
	std::map<std::string, DataSet*> m_mCatalogue;

	/**
	 * SVM training data, part of catalogue data excluding validation data
	 */
	std::map<std::string, TrainingData*> m_mTrain;

	/**
	 * SVM validation data, part of catalogue data excluding training data
	 */
	std::map<std::string, TrainingData*> m_mValidate;

	/**
	 * Trained SVMs per named category
	 */
	std::map<std::string, umSVM*> m_mSVMs;

	/**
	 * Application settings
	 */
	Configuration*	m_Config;
};

}// namespace mai



#endif /* SRC_CATALOGUEDETECTION_H_ */
