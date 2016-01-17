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
#include <vector>

namespace mai{

class DataSet;

/**
 * OpenCV Support Vector Machine application
 *
 * @see http://docs.opencv.org/2.4/modules/ml/doc/support_vector_machines.html#
 */
class umSVM
{
public:
	/**
	 * Initializes object with new OpenCV SVM.
	 *
	 * @param dCValue	Penalty multiplier for outliers on imperfect separation.
	 */
	umSVM(double dCValue = 0.1);

	/**
	 * Deletes CvSVM
	 */
	virtual ~umSVM();

	/**
	 * Train linear SVM with standard parameters.
	 *
	 * @param[in] data		Training data
	 * @param[in] labels	Labels
	 * @param[out] vSupport	Support vectors
	 * @return				Number of support vectors
	 */
	int trainSVM(const cv::Mat &data,
			const cv::Mat &labels,
			std::vector<std::vector<float> > &vSupport);

	/**
	 * Predict the given image.
	 *
	 * @param data				Input image
	 * @param bReturnfDFValue	Return DFValue or label ? (@see OpenCV documentation)
	 * @return					Label(false) or value depending on bReturnfDFValue
	 */
	float predict(const cv::Mat &data, bool bReturnfDFValue = false);

	/**
	 * Predict the given data.
	 *
	 * @param data		Input matrix, images row-wise
	 * @param results	Prediction results corresponding row-wise ( labels of type float )
	 */
	void predict(const cv::Mat &data, cv::Mat &results);

	/**
	 * Save svm as xml. ".xml" will be added to the given name.
	 *
	 * @param strFilename	Filename to save the SVM at.
	 */
	void saveSVM(const std::string &strFilename);

	/**
	 * Load svm from given file.
	 *
	 * @param strFilename	Filename of trained SVM to load.
	 */
	void loadSVM(const std::string &strFilename);

	/**
	 * Search vectors of vSupport in input data vData.
	 *
	 * @param vData		SVM training data
	 * @param vSupport	SVM support vectors
	 * @param bSort		Should the vectors be sorted before comparison ?
	 */
	static void searchSupportVector(const std::vector<std::vector<float> > &vData,
				const std::vector<std::vector<float> > &vSupport,
				bool bSort = false);

	/**
	 * Search vectors of vSupport in input data data.
	 *
	 * @param data		SVM training data
	 * @param vSupport	SVM support vectors
	 * @param bSort		Should the vectors be sorted before comparison ?
	 */
	static void searchSupportVector(const DataSet* const data,
			const std::vector<std::vector<float> > &vSupport,
			bool bSort = false);

private:

	CvSVM*	m_pSVM;
	double	m_dCValue;

};

}// namespace mai

#endif /* SRC_SVM_UMSVM_H_ */
