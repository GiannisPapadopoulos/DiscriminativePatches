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

#ifndef SRC_SVM_CATALOGUETRAINING_H_
#define SRC_SVM_CATALOGUETRAINING_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

class Configuration;
class DataSet;
class TrainingData;
class ClassificationSVM;

/**
 * Train classifiers and predict sampled images on a catalogue of labeled images.
 *
 */
class CatalogueTraining
{
public:

	/**
	 * Load image catalogue according to configuration
	 */
	CatalogueTraining(Configuration* config);

	/**
	 * Deletes everything
	 */
	virtual ~CatalogueTraining();

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
	 * Extract faces from all images and replace images by discovered faces for further processing.
	 *
	 * @see utils/FaceDetection::detectFaces
	 */
	void detectFaces();

	/**
	 * Computes HOG descriptors for all Datasets in catalogue.
	 * @see featureExtraction/umHOG::computeHOGForDataSet
	 *
	 * After feature extraction images are removed from dataset to free memory.
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
	 * k-means clustering.
	 * First  patches are extracted from feature vectors.
	 */
	void performClustering(cv::Size imageSize,
			cv::Size blockSize,
			cv::Size blockStride,
			cv::Size cellSize,
			int iNumBins);

	/**
	 * Setup training data for support vector machine.
	 * Training and validation data is split according to dataset divider.
	 * For each category in the catalogue a training matirx is constructed using the images of that category as positive samples
	 * and the same amount of images taken evenly from all other categries as negative samples.
	 * Negative samples inside a category are chsoen randomly.
	 *
	 * @param iDataSetDivider	divider of dataset size defining validation part, e.g. 4 -> 1/4 of patches will be in validation set.
	 */
	void setupSVMData(int iDataSetDivider = 1);

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
	 */
	void collectRandomNegatives(std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives);

	/**
	 * Calculate exact negative sample sizes for a category taking divison rest and partially insufficient samples in one or more categories into account.
	 *
	 * @param strKey			key of the currently processed category
	 * @param[in] mFeatureSizes	Map containing the existing sample sizes for all categories
	 * @param[out] mSampleSizes	Map where the calculated sample sizes get stored
	 */
	void calculateSampleSizes(std::string strKey,
			std::map<std::string, int> &mFeatureSizes,
			std::map<std::string, int> &mSampleSizes);

	/**
	 * Creates training data from positive and negative samples.
	 * @see data/TrainingData
	 */
	void setupTrainingData(std::map<std::string, TrainingData*> &mTrain,
			std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives);


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
	 * Trained svms
	 */
	ClassificationSVM* m_Classifiers;

	/**
	 * Application settings
	 */
	Configuration*	m_Config;
};

}// namespace mai



#endif /* SRC_SVM_CATALOGUETRAINING_H_ */