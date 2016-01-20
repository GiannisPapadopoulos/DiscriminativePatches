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

#ifndef SRC_IMAGECATALOG_CATALOGTRAINING_H_
#define SRC_IMAGECATALOG_CATALOGTRAINING_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

class Configuration;
class DataSet;
class TrainingData;
class CatalogClassificationSVM;

/**
 * Train classifiers and predict sampled images on a cataloge of labeled images.
 *
 */
class CatalogTraining
{
public:

	/**
	 * Load image catalogue according to configuration
	 */
	CatalogTraining(const Configuration* const config);

	/**
	 * Deletes everything
	 */
	virtual ~CatalogTraining();

	/**
	 * Processing pipeline for extracted patches:
	 * - Refine images though histogram equalization
	 * - Compute HOG features
	 * - Setup training and validation data
	 * - Train SVMs
	 * - Predict data
	 */
	void processPipeline();

private:

	/**
	 * Extract faces from all images and replace images in the cataloge datasets by discovered faces for further processing.
	 *
	 * @see utils/FaceDetection::detectFaces
	 */
	void detectFaces();

	/**
	 * Computes HOG descriptors for all datasets in cataloge.
	 * @see featureExtraction/umHOG::computeHOGForDataSet
	 *
	 * After feature extraction images are removed from dataset to free memory.
	 *
	 * @param imageSize			Images will be resized to this size.
	 * @param blockSize			HOG feature extraction block size.
	 * @param blockStride		HOG feature extraction block stride.
	 * @param cellSize			HOG feature extraction cell size.
	 * @param iNumBins			HOG feature extraction gradient directions.
	 * @param winstride			HOG feature extraction window stride.
	 * @param padding			HOG feature extraction padding.
	 * @param bWriteHOGImages	Write HOG visualization images to disc ?
	 * @param bApplyPCA			Apply Principal Component Analysis to feature vectors ?
	 */
	void computeHOG(const cv::Size imageSize,
				const cv::Size blockSize,
				const cv::Size blockStride,
				const cv::Size cellSize,
				const int iNumBins,
				const cv::Size winStride,
				const cv::Size padding,
				const bool bWriteHOGImages = false,
				const bool bApplyPCA = false);


	/**
	 * Setup training data for support vector machine.
	 * Training and validation data is split according to dataset divider.
	 * For each category in the cataloge a training matirx is constructed using the images of that category as positive samples
	 * and an equal amount of images taken evenly from all other categories as negative samples.
	 * Negative samples inside a category are chosen randomly.
	 *
	 * @param iDataSetDivider	divider of dataset size defining validation part, e.g. 4 -> 1/4 of patches will be in validation set.
	 */
	void setupSVMData(const int iDataSetDivider = 1);

	/**
	 * Assigns part of the descriptor vectors of each dataset for validation purpose, the rest for training.
	 *
	 * @param[out] mTrain		Training part of cataloge data.
	 * @param[out] mValidate	Validation part of cataloge data.
	 * @param iDataSetDivider	divider of dataset size defining validation part, e.g. 4 -> 1/4 of patches will be in validation set.
	 */
	void divideDataSets(std::map<std::string, std::vector<std::vector<float> > > &mTrain,
			std::map<std::string, std::vector<std::vector<float> > > &mValidate,
			const int iDataSetDivider);

	/**
	 * Collects random samples for each positive descriptor set from all other descriptor sets evenly distributed.
	 * Size of the sampled descriptor sets are the same as the positive one to have an equal amount of positive and negative training samples.
	 * Negative samples inside a category are chosen randomly.
	 *
	 * @param[in] mPositives	Map of samples from the corresponding cataloge categories.
	 * @param[out] mNegatives	Map of collected samples from all other categories organized per original category.
	 */
	void collectRandomNegatives(const std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives);

	/**
	 * Calculate exact negative sample sizes for a category taking division rest and partially insufficient samples in one or more categories into account.
	 *
	 * @param strKey			key of the currently processed category.
	 * @param[in] mFeatureSizes	Map containing the existing sample sizes for all categories.
	 * @param[out] mSampleSizes	Map where the calculated sample sizes get stored.
	 */
	void calculateSampleSizes(const std::string &strKey,
			const std::map<std::string, int> &mFeatureSizes,
			std::map<std::string, int> &mSampleSizes);

	/**
	 * Creates training data from positive and negative samples.
	 * @see data/TrainingData
	 *
	 * @param[out] mTrain	Map of constructed training datasets per category.
	 * @param[in] mPositives	Map of positive training samples per category.
	 * @param[in] mNegatives	Map of negative training samples per category.
	 */
	void setupTrainingData(std::map<std::string, TrainingData*> &mTrain,
			const std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			const std::map<std::string, std::vector<std::vector<float> > > &mNegatives);

	/**
	 * Images and extracted feature vectors per named category
	 */
	std::map<std::string, DataSet*> m_mCatalogue;

	/**
	 * SVM training data, part of cataloge data excluding validation data
	 */
	std::map<std::string, TrainingData*> m_mTrain;

	/**
	 * SVM validation data, part of cataloge data excluding training data
	 */
	std::map<std::string, TrainingData*> m_mValidate;

	/**
	 * Trained svms
	 */
	CatalogClassificationSVM* m_Classifiers;

	/**
	 * Application settings
	 */
	const Configuration* const	m_Config;
};

}// namespace mai



#endif /* SRC_IMAGECATALOG_CATALOGTRAINING_H_ */
