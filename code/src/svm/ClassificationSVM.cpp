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
#include "../Configuration.h"
#include "../featureExtraction/umHOG.h"

#include <opencv2/highgui/highgui.hpp>

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

string mai::ClassificationSVM::predict(const Mat &image,
		map<string, float> &mResults,
		Configuration* config)
{
	if(m_mSVMs.size() < 1)
	{
		cout << "[mai::ClassificationSVM::predict] ERROR! No traind svms." << endl;
		return "";
	}

	Size cellSize = config->getCellSize();
	Size blockStride = config->getBlockStride();
	Size blockSize = config->getBlockSize();
	Size imageSize = config->getImageSize();

	// Not sure about these
	Size winStride = Size(0,0);
	Size padding = Size(0,0);

	int iNumBins = config->getNumBins();

	vector<float> descriptorsValues;
	umHOG::extractFeatures(descriptorsValues,
			image,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			iNumBins,
			winStride,
			padding);

	// setup matrix
	Mat predictionData(1, descriptorsValues.size(), CV_32FC1, &descriptorsValues[0]);;
	//		for(unsigned int j = 0; j < descriptorsValues.size(); ++j)
	//		{
	//			predictionData.at<float>(0, j) = descriptorsValues[j];
	//		}

	if(Constants::DEBUG_SVM_PREDICTION)
	{
		cout << "[mai::ClassificationSVM::predict] Prediction matrix " << predictionData.rows << "x" << predictionData.cols << endl;
	}

	float fBestResultValue = std::numeric_limits<float>::max();
	string strBest = "";
	for(map<string, umSVM*>::iterator itSVMs = m_mSVMs.begin(); itSVMs != m_mSVMs.end(); itSVMs++)
	{
		string strCategory = itSVMs->first;
		umSVM* svm = itSVMs->second;
		float fResultLabel = svm->predict(predictionData, false);
		float fResultValue = svm->predict(predictionData, true);

		if(Constants::DEBUG_SVM_PREDICTION)
		{
		    cout << "[mai::ClassificationSVM::predict] SVM prediction in category " << strCategory << " is " << fResultLabel << ", DFvalue " << fResultValue << endl;
		}

		if(fResultValue < fBestResultValue)
		{
			fBestResultValue = fResultValue;
			strBest = strCategory;
		}

		mResults.insert(pair<string, float>(strCategory, fResultValue));
	}

	if(Constants::DEBUG_SVM_PREDICTION)
	{
	    cout << "[mai::ClassificationSVM::predict] Best prediction for given image is " << strBest << endl;
	}

	return strBest;
}

void mai::ClassificationSVM::loadAndPredictImage(const string &strFilename,
		Configuration* config)
{
	cv::Mat image;
	if(!IOUtils::loadImage(image,
			IMREAD_GRAYSCALE,
			strFilename,
			true))
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] ERROR! Loading image from " << strFilename << endl;
		return;
	}

	string strSVMPath = config->getSvmOutputPath();

	if(Constants::DEBUG_SVM_PREDICTION)
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] Loading classifiers from " << strSVMPath << endl;
	}

	loadSVMs(strSVMPath);

	map<string, float> mResults;
	string strClassifiedAs = predict(image,
			mResults,
			config);

	cout << "#-------------------------------------------------------------------------------#" << endl;
	cout << "[mai::ClassificationSVM::loadAndPredictImage] Prediction done. Best classification is " << strClassifiedAs << endl;
	cout << "#-------------------------------------------------------------------------------#" << endl;
}

