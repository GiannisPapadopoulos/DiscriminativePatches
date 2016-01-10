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

	bool loadSVMs(const std::string &strPath);

	void predict(std::map<std::string, TrainingData*> &mData,
			std::map<std::string, cv::Mat> &mResults);

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
