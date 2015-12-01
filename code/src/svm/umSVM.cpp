/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * cvSVM.cpp
 *
 *  Created on: Nov 24, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "umSVM.h"
#include "../Constants.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>

using namespace cv;

using namespace std;


mai::umSVM::umSVM()
	: m_pSVM(new CvSVM())
{}

mai::umSVM::~umSVM()
{
	delete m_pSVM;
}

int mai::umSVM::trainSVM(Mat &data,
		Mat &labels,
		std::vector<float> &vSupport)
{
	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = Constants::SVM_C_VALUE;
//	params.gamma = 3;
//	params.degree = 3;
//	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	cout << "[mai::umSVM::trainSVM] training svm .." << endl;

	// Train the SVM
	m_pSVM->train(data, labels, Mat(), Mat(), params);

	int numSupportVectors = m_pSVM->get_support_vector_count();

	vSupport.clear();
	for(int i = 0; i < numSupportVectors; ++i)
	{
		vSupport.push_back(*(m_pSVM->get_support_vector(i)));
	}

	cout << "[mai::umSVM::trainSVM] svm trained, support vector count: " << numSupportVectors << endl;

	return numSupportVectors;
}

float mai::umSVM::predict(Mat &data, bool bReturnDFValue)
{
	float fResult = m_pSVM->predict(data, bReturnDFValue);
	return fResult;
}

void mai::umSVM::saveSVM(std::string &strFilename)
{
	m_pSVM->save(strFilename.c_str());
}

void mai::umSVM::loadSVM(std::string &strFilename)
{
	m_pSVM->load(strFilename.c_str());
}
