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

class DataSet;
class TrainingData;
class umSVM;

/**
 * Unsupervised discovery of mid-level discriminative patches
 *
 */
class CatalogueDetection
{
public:
	/**
	 * Initializes object
	 */
	CatalogueDetection();

	/**
	 * Deletes something
	 */
	virtual ~CatalogueDetection();

	/**
	 *
	 */
	void catalogueDetection(std::string &strFilePath);

private:

	/**
	 *
	 */
	void computeHOGForCatalogue(std::map<std::string, DataSet*> &mCatalogue,
				cv::Size imageSize,
				cv::Size blockSize,
				cv::Size blockStride,
				cv::Size cellSize,
				cv::Size winStride,
				cv::Size padding);

	/**
	 * Computes features for all images in data.
	 * Adds them as DescriptorValues to the corresponding image in the DataSet.
	 * @see class DataSet::ImageWithDescriptors
	 *
	 * Method used:
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
	 * @see svm/umSVM::train
	 */
	void trainSVMsForCatalogue(std::map<std::string, DataSet*> &mCatalogue);

	void collectPositivesFromCatalogue(std::map<std::string, DataSet*> &mCatalogue,
			std::map<std::string, std::vector<std::vector<float> > > &mTrain,
			std::map<std::string, std::vector<std::vector<float> > > &mValidate);

	void collectRandomNegatives(std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives);

	void setupTrainingData(std::map<std::string, TrainingData*> &m_mTrain,
			std::map<std::string, DataSet*> &mCatalogue,
			std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives);


	std::map<std::string, DataSet*> m_mCatalogue;

	std::map<std::string, TrainingData*> m_mTrain;
	std::map<std::string, TrainingData*> m_mValidate;

	std::map<std::string, umSVM*> m_mSVMs;

};

}// namespace mai



#endif /* SRC_CATALOGUEDETECTION_H_ */
