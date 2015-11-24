/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * cvSVM.h
 *
 *  Created on: Nov 24, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_SVM_UMSVM_H_
#define SRC_SVM_UMSVM_H_

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <string>

namespace mai{

/**
 * Doxygen comments
 * Class description
 */
class umSVM
{
public:
	/**
	 * Initializes object
	 */
	umSVM();

	/**
	 * Deletes something
	 */
	virtual ~umSVM();

	/**
	 * Train with standard parameters
	 * @param data		training data
	 * @param labels	labels
	 * @param[out] vSupport	support vectors
	 * @return			number of support vectors
	 */
	int trainSVM(cv::Mat &data,
			cv::Mat &labels,
			std::vector< float> &vSupport);

	/**
	 * Predict the given image
	 * @param data		input image
	 * @param bReturnfDFValue	return DFValue or label (@see OpenCV documentation)
	 * @return	label(false) or value depending on bReturnfDFValue
	 */
	float predict(cv::Mat &data, bool bReturnfDFValue);

	/**
	 *
	 */
	void saveSVM(std::string &strFilename);

	/**
	 *
	 */
	void loadSVM(std::string &strFilename);


private:

	CvSVM*	m_pSVM;

};

}// namespace mai

#endif /* SRC_SVM_UMSVM_H_ */