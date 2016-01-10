/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * ImageClassification.h
 *
 *  Created on: Jan 10, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_SVM_CLASSIFICATIONSVM_H_
#define SRC_SVM_CLASSIFICATIONSVM_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

class TrainingData;
class umSVM;
class Configuration;

/**
 * Unsupervised discovery of mid-level discriminative patches
 *
 */
class ClassificationSVM
{
public:

	/**
	 * Constructor
	 */
	ClassificationSVM();

	/**
	 * Deletes classifiers
	 */
	virtual ~ClassificationSVM();

	/**
	 * Train svms.
	 * @see svm/umSVM::train
	 *
	 * Saves trained svms by catalogue category labels.
	 */
	void trainSVMs(std::map<std::string, TrainingData*> &mTrainingData,
			double dCValue);

	/**
	 * Load trained svms from filesystem.
	 *
	 * @param strPath	Path to folder containing traind svm descriptions.
	 * @return			Were there any errors on loading ?
	 */
	bool loadSVMs(const std::string &strPath);


	/**
	 * Predict data using the stored svms.
	 *
	 * @param[in] mData		Data that has to be predicted.
	 * @param[out] mResults	Result matrix containing predicted labels according to input matrices
	 */
	void predict(std::map<std::string, TrainingData*> &mData,
			std::map<std::string, cv::Mat> &mResults);

	/**
	 * Classify a single image by the stored svms.
	 *
	 * @param[in] image		Image to be classified
	 * @param[out] mResults Predicted distances for each svm category
	 * @param config		Configuration containing HOG parameters as used for training
	 * @return				Category name of best prediction
	 */
	std::string predict(const cv::Mat &image,
			std::map<std::string, float> &mResults,
			Configuration* config);

	void loadAndPredictImage(const std::string &strFilename,
			Configuration* config);

	/**
	 * Access the stored svms
	 */
	const std::map<std::string, umSVM*>& getSVMs() const {
		return m_mSVMs;
	}

private:

	/**
	 * Trained SVMs per named category
	 */
	std::map<std::string, umSVM*> m_mSVMs;

};

}// namespace mai

#endif /* SRC_SVM_CLASSIFICATIONSVM_H_ */
