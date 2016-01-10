/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * ImageClassification.cpp
 *
 *  Created on: Jan 10, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "ClassificationSVM.h"

#include "umSVM.h"
#include "../IO/IOUtils.h"
#include "../Constants.h"
#include "../data/TrainingData.h"

using namespace std;
using namespace cv;

mai::ClassificationSVM::ClassificationSVM()
{}

mai::ClassificationSVM::~ClassificationSVM()
{
	for(map<string, umSVM*>::iterator it = m_mSVMs.begin(); it != m_mSVMs.end(); it++)
	{
		delete it->second;
	}
	m_mSVMs.clear();
}

void mai::ClassificationSVM::trainSVMs(map<string, TrainingData*> &mTrainingData,
		double dCValue)
{
	// Train svms for each category
	for(map<string, TrainingData*>::const_iterator it = mTrainingData.begin(); it != mTrainingData.end(); it++)
	{
		string strName = it->first;
		TrainingData* data = it->second;
		umSVM* svm = new umSVM(dCValue);
		vector<vector<float> > vSupport;

//		std::string strDataname = "trainingdata";
//		IOUtils::writeMatToCSV(data->getData(), strDataname);
//		std::string strLabelname = "labeldata";
//		IOUtils::writeMatToCSV(data->getLabels(), strLabelname);

		svm->trainSVM(data->getData(), data->getLabels(), vSupport);

		m_mSVMs.insert(pair<string, umSVM*>(strName, svm));

		if(Constants::DEBUG_SVM)
		{
			cout << "[mai::ClassificationSVM::trainSVMs] svm traind for " << strName << endl;
		}
	}
}

bool mai::ClassificationSVM::loadSVMs(const std::string &strPath)
{
	if(!IOUtils::loadSVMsFromDirectory(m_mSVMs, strPath))
	{
		return false;
	}

	return true;
}

void mai::ClassificationSVM::predict(map<string, TrainingData*> &mData,
		map<string, Mat> &mResults)
{
	if(m_mSVMs.size() < 1)
	{
		cout << "[mai::ClassificationSVM::predict] ERROR! No traind svms." << endl;
		return;
	}

	// Predict data by trained svms for each category
	for(map<string, TrainingData*>::const_iterator it = mData.begin();
			it != mData.end(); it++)
	{
		string strName = it->first;
		umSVM* svm;
		try
		{
			svm = m_mSVMs.at(strName);
		}
		catch (out_of_range &e)
		{
			cout << "[mai::ClassificationSVM::predict] ERROR! No trained svm of name " << strName << endl;
			return;
		}

		// result matrix containing predicted labels
		Mat results(it->second->getData().rows, 1, CV_32SC1);

		svm->predict(it->second->getData(), results);

		int iResultsRows = results.rows;
		int iCorrectDetection = 0;

		cout << "[mai::ClassificationSVM::predict] SVM prediction result for label " << strName << " has " << iResultsRows << " rows." << endl;

		// Compare predicted labels with original ones
		for(int i = 0; i < iResultsRows; ++i)
		{
			int iResultLabel = results.at<float>(i);
			int iDataLabel = it->second->getLabels().at<int>(i);

			if(iResultLabel == iDataLabel)
				iCorrectDetection++;

			if(Constants::DEBUG_SVM_PREDICTION)
				cout << "[mai::ClassificationSVM::predict] SVM predict for image " << i << " is " << iResultLabel << ". Label was : " << iDataLabel << endl;
		}

		double dCorrectDetectionRatio = (double)iCorrectDetection / iResultsRows;
		cout << "[mai::ClassificationSVM::predict] Ratio of correct detections for label " << strName << " : " << dCorrectDetectionRatio << endl;

		mResults.insert(pair<string, Mat>(strName, results));
	}
}

