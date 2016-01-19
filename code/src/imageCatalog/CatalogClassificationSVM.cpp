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

#include "CatalogClassificationSVM.h"

#include "../svm/umSVM.h"
#include "../IO/IOUtils.h"
#include "../data/TrainingData.h"
#include "../featureExtraction/umHOG.h"
#include "../configuration/Configuration.h"
#include "../configuration/Constants.h"

#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

mai::CatalogClassificationSVM::CatalogClassificationSVM(const Configuration* const config)
:	m_Config(config)
{}

mai::CatalogClassificationSVM::~CatalogClassificationSVM()
{
	for(map<string, umSVM*>::iterator it = m_mSVMs.begin(); it != m_mSVMs.end(); it++)
	{
		delete it->second;
	}
	m_mSVMs.clear();
}

void mai::CatalogClassificationSVM::trainSVMs(const map<string, TrainingData*> &mTrainingData,
		const double dCValue)
{
	// Train svms for each category
	for(map<string, TrainingData*>::const_iterator it = mTrainingData.begin(); it != mTrainingData.end(); it++)
	{
		string strName = it->first;
		TrainingData* data = it->second;
		vector<vector<float> > vSupport;
		umSVM* svm;

		try
		{
			svm = m_mSVMs.at(strName);
		}
		catch (out_of_range &e)
		{
			svm = new umSVM(dCValue);
		}

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

bool mai::CatalogClassificationSVM::loadSVMs(const std::string &strPath)
{
	if(!IOUtils::loadSVMsFromDirectory(m_mSVMs, strPath))
	{
		return false;
	}

	return true;
}

void mai::CatalogClassificationSVM::predict(const map<string, TrainingData*> &mData,
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

		map<string, Mat>::iterator itFound = mResults.find(strName);
		if (itFound != mResults.end())
		{
			mResults.erase(itFound);
		}

		mResults.insert(pair<string, Mat>(strName, results));
	}
}

string mai::CatalogClassificationSVM::predict(const Mat &image,
		map<string, float> &mResults)
{
	if(m_mSVMs.size() < 1)
	{
		cout << "[mai::ClassificationSVM::predict] ERROR! No traind svms." << endl;
		return "";
	}

	Size cellSize = m_Config->getCellSize();
	Size blockStride = m_Config->getBlockStride();
	Size blockSize = m_Config->getBlockSize();
	Size imageSize = m_Config->getImageSize();

	// Not sure about these
	Size winStride = Size(0,0);
	Size padding = Size(0,0);

	int iNumBins = m_Config->getNumBins();

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

void mai::CatalogClassificationSVM::loadAndPredict()
{
	string strFilename = m_Config->getImageInputPath();
	string strSVMPath = m_Config->getSvmOutputPath();

	if(Constants::DEBUG_SVM_PREDICTION)
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] Loading classifiers from " << strSVMPath << endl;
	}

	if(!loadSVMs(strSVMPath))
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] ERROR! Loading svms " << strSVMPath << endl;
		return;
	}

	if(IOUtils::getIsDirectory(strFilename))
	{
		loadAndPredictImages();
	}
	else if(IOUtils::getIsFile(strFilename))
	{
		loadAndPredictImage();
	}
	else
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] ERROR! File or directory does not exist: " << strFilename << endl;
	}
}

void mai::CatalogClassificationSVM::loadAndPredictImages()
{
	string strFilename = m_Config->getImageInputPath();

	vector<Mat*> vImages;
	vector<string> vImageNames;

	if(!IOUtils::loadImagesOrdered(vImages,
			vImageNames,
			IMREAD_GRAYSCALE,
			strFilename,
			true))
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] ERROR! Loading image from " << strFilename << endl;
		return;
	}

	if(vImageNames.size() != vImages.size())
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] ERROR! Loading image from " << strFilename << endl;
		return;
	}

	for(unsigned int i = 0; i < vImages.size(); ++i)
	{
		Mat* image = vImages.at(i);
		string strName = vImageNames.at(i);

		predictAndShowResults(*image, strName);
	}
}

void mai::CatalogClassificationSVM::loadAndPredictImage()
{
	string strFilename = m_Config->getImageInputPath();

	Mat image;
	if(!IOUtils::loadImage(image,
			IMREAD_GRAYSCALE,
			strFilename,
			true))
	{
		cout << "[mai::ClassificationSVM::loadAndPredictImage] ERROR! Loading image from " << strFilename << endl;
		return;
	}

	predictAndShowResults(image, strFilename);
}

void mai::CatalogClassificationSVM::predictAndShowResults(const Mat &image,
		const string &strName)
{
	map<string, float> mResults;

	string strClassifiedAs = predict(image, mResults);

	cout << "#-------------------------------------------------------------------------------#" << endl;
	cout << " Prediction for image " << strName << endl;
	cout << " Best classification is " << strClassifiedAs << endl;
	cout << "#-------------------------------------------------------------------------------#" << endl;
}
