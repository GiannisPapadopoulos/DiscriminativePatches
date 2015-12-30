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
#include "../data/DataSet.h"
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

int mai::umSVM::trainSVM(const Mat &data,
		const Mat &labels,
		vector<vector<float> > &vSupport)
{
	// Set up SVM's parameters
	CvSVMParams params;

	params.svm_type    = CvSVM::C_SVC;
	params.C = Constants::SVM_C_VALUE;

	params.kernel_type = CvSVM::LINEAR;

//	params.kernel_type = CvSVM::RBF;
//	params.gamma = 0.1;

//	params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

	if(Constants::DEBUG_SVM)
		cout << "[mai::umSVM::trainSVM] training svm .." << endl;

	// Train the SVM
	m_pSVM->train(data, labels, Mat(), Mat(), params);

	int numSupportVectors = m_pSVM->get_support_vector_count();

	vSupport.clear();
	int featureSize = data.cols;
	for(int i = 0; i < numSupportVectors; ++i)
	{
		const float* supportVector = m_pSVM->get_support_vector(i);

		vector<float> temp;
		for (int j = 0; j < featureSize; ++j) {
			temp.push_back(supportVector[j]);
		}
		vSupport.push_back(temp);
	}

	if(Constants::DEBUG_SVM)
		cout << "[mai::umSVM::trainSVM] svm trained, support vector count: " << numSupportVectors << " with feature size " << featureSize << endl;

	return numSupportVectors;
}

float mai::umSVM::predict(const Mat &data, bool bReturnDFValue)
{
	float fResult = m_pSVM->predict(data, bReturnDFValue);
	return fResult;
}

void mai::umSVM::predict(const Mat &data, Mat &results)
{
	m_pSVM->predict(data, results);
}

void mai::umSVM::saveSVM(const string &strFilename)
{
	stringstream sstm;
	sstm << strFilename << ".xml";

//#ifdef linux
	m_pSVM->save(sstm.str().c_str());
//#else
//	m_pSVM->save(sstm.str().string());
//#endif
}

void mai::umSVM::loadSVM(const string &strFilename)
{
//#ifdef linux
	m_pSVM->load(strFilename.c_str());
//#else
//	m_pSVM->load(strFilename.string());
//#endif
}

void mai::umSVM::searchSupportVector(vector<vector<float> > &vData,
			vector<vector<float> > &vSupport,
			bool bSort)
{
	for(unsigned int i = 0; i < vSupport.size(); ++i)
	{
		vector<float> temp = vSupport[i];
		if(bSort)
			sort(temp.begin(), temp.end());

		for(unsigned int j = 0; j < vData.size(); ++j)
		{
			vector<float> desc = vData.at(j);

			if(bSort)
				sort(desc.begin(), desc.end());

			if(desc == temp)
			{
				cout << "[mai::umSVM::searchSupportVector] Support Vector match at DateSet index " << j << endl;
			}
		}
	}
}

void mai::umSVM::searchSupportVector(DataSet* data,
			vector<vector<float> > &vSupport,
			bool bSort)
{
	for(unsigned int i = 0; i < vSupport.size(); ++i)
	{
		vector<float> temp = vSupport[i];
		if(bSort)
			sort(temp.begin(), temp.end());

		for(unsigned int j = 0; j < data->getImageCount(); ++j)
		{
			vector<float> desc;
			data->getDescriptorValuesFromImageAt(j, desc);

			if(bSort)
				sort(desc.begin(), desc.end());

			if(desc == temp)
			{
				cout << "[mai::umSVM::searchSupportVector] Support Vector match at DateSet index " << j << endl;
			}
		}
	}
}
