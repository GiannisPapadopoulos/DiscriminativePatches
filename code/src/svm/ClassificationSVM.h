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
 * Holds map of categorized SVMs and provides necessary methods to interact with them.
 */
class ClassificationSVM
{
public:

	/**
	 * Constructor
	 */
	ClassificationSVM(const Configuration* const config);

	/**
	 * Deletes classifiers
	 */
	virtual ~ClassificationSVM();

	/**
	 * Train SVMs for each given category.
	 * @see svm/umSVM::train
	 *
	 * Saves trained SVMs by catalogue category labels.
	 *
	 * @param[in] mTrainingData	Training and label matrices.
	 * @param dcValue			Linear SVM c value.
	 */
	void trainSVMs(const std::map<std::string, TrainingData*> &mTrainingData,
			const double dCValue);

	/**
	 * Load trained SVMs from filesystem.
	 *
	 * @param strPath	Path to folder containing traind svm descriptions.
	 * @return			Were there any errors on loading ?
	 */
	bool loadSVMs(const std::string &strPath);


	/**
	 * Predict data using the stored SVMs.
	 *
	 * @param[in] mData		Data that has to be predicted.
	 * @param[out] mResults	Result matrix containing predicted labels according to input matrices
	 */
	void predict(const std::map<std::string, TrainingData*> &mData,
			std::map<std::string, cv::Mat> &mResults);

	/**
	 * Classify a single image or a folder of images by the stored SVMs.
	 *
	 * @param[in] image		Image to be classified
	 * @param[out] mResults Predicted distances for each svm category
	 * @return				Category name of best prediction
	 */
	std::string predict(const cv::Mat &image,
			std::map<std::string, float> &mResults);

	/**
	 * Load and predict images according to configuration.
	 */
	void loadAndPredict();

	/**
	 * @return	The stored and trained SVMs.
	 */
	const std::map<std::string, umSVM*>& getSVMs() const {
		return m_mSVMs;
	};

private:

	void loadAndPredictImage();

	void loadAndPredictImages();

	void predictAndShowResults(const cv::Mat &image,
			const std::string &strName);

	/**
	 * Trained SVMs per named category
	 */
	std::map<std::string, umSVM*> m_mSVMs;

	/**
	 * Application settings
	 */
	const Configuration* const	m_Config;
};

}// namespace mai

#endif /* SRC_SVM_CLASSIFICATIONSVM_H_ */
