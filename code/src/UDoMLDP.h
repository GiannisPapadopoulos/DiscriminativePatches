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
 *
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
	 * Loads positive and negative images and splits them by half.
	 * The members m_pPositiveTrain, m_pNegativeTrain, m_pPositiveValid and
	 * m_pNegativeValid are then filled with the images and ready for further use.
	 *
	 * Next HOG features are computed for each image.
	 * Adjust the size of blocks, strides and cells in code !
	 * The method called is computeHOGForDataSet
	 *
	 * Next the svm is trained with the training datasets.
	 * Method called: trainSVMOnDataSets.
	 *
	 * Then the prediction is done with the validation datasets.
	 */
	void basicDetecion(std::string &strFilePathPositives, std::string &strFilePathNegatives);

private:

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
	 * Predict dataset
	 * @see svm/umSVM::predict
	 */
	void predictDataSetbySVM(DataSet* data);

	/**
	 * Calls methods to setup training data:
	 * Either setupTrainingData or setupTrainingDataForSinglePatchImage can be used.
	 * Check and adjust code as needed !
	 *
	 * Then svm is trained using:
	 * @see svm/umSVM::train
	 */
	void trainSVMOnDataSets(DataSet* positives,
			DataSet* negatives);

	void searchSupportVector(DataSet* data,
			std::vector<std::vector<float> > vSupport);

	/**
	 * Contruct Mat for trainingdata and labels used by svm.
	 * Calls collectTrainingDataAndLabels for positives and negatives
	 * Training data are the single floats of the feature vectors !
	 */
	void setupTrainingData(DataSet* positives,
			DataSet* negatives,
			cv::Mat &trainingData,
			cv::Mat &labels);
	/**
	 * Put the descriptorValues from the DataSet into TrainingData and setup labels like given
	 * @param[in] data				positive or negative training data
	 * @param[out] vTrainingData	training data vector
	 * @param[out] vLabel			labels vector
	 * @param fLabel				labels for this set, e.g. 1.0 if positive or 0.0 if negative
	 */
	void collectTrainingDataAndLabels(DataSet* data,
			std::vector<float> &vTrainingData,
			std::vector<float> &vLabels,
			float fLabel);


	/**
	 * Contruct Mat for trainingdata and labels used by svm.
	 * Calls collectTrainingDataAndLabelsForSingelPatchImage for positives and negatives.
	 * Training data is the whole feature vector of an image ( image = patch ).
	 * There should be number of image rows in the training matrix and number of features columns.
	 */
	void setupTrainingDataForSinglePatchImage(DataSet* positives,
				DataSet* negatives,
				cv::Mat &trainingData,
				cv::Mat &labels);

	/**
	 * Put the descriptorValues from the DataSet into TrainingData and setup labels like given
	 */
	int collectTrainingDataAndLabelsForSingelPatchImage(DataSet* data,
			std::vector<std::vector<float> > &vTrainingData,
			std::vector<float> &vLabels,
			float fLabel);

	/**
	 * Convert each feature vector to matrix and do svm predict on that.
	 */
	void predictDataSetbySVMForSinglePatchImage(DataSet* data);

	void predictWholeDataSetbySVMForSinglePatchImage(DataSet* data);

	// Training datasets
	DataSet*	m_pPositiveTrain;
	DataSet*	m_pNegativeTrain;
	// Validation datasets
	DataSet*	m_pPositiveValid;
	DataSet*	m_pNegativeValid;

	// trainable svm
	umSVM*		m_pSVM;

};

}// namespace mai

#endif /* SRC_UDOMLDP_H_ */
