/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * CatalogueDetection.cpp
 *
 *  Created on: Dec 28, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "CatalogueDetection.h"
#include "Constants.h"

#include "svm/umSVM.h"
#include "data/DataSet.h"
#include "data/TrainingData.h"
#include "IO/IOUtils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include "featureExtraction/umHOG.h"


using namespace cv;

using namespace std;


mai::CatalogueDetection::CatalogueDetection(std::string &strFilePath)
{
	IOUtils::loadCatalogue(m_mCatalogue, IMREAD_COLOR, strFilePath, true);

	cout << "[mai::CatalogueDetection::catalogueDetection] Number of labels: " << m_mCatalogue.size() << endl;
	for (map<string, DataSet*>::const_iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); ++it)
	{
		cout << "[mai::CatalogueDetection::catalogueDetection] Label: " << it->first << ", number of images: " << it->second->getImageCount() << endl;
	}
}

mai::CatalogueDetection::~CatalogueDetection()
{
	for(map<string, DataSet*>::iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		delete it->second;
	}
	m_mCatalogue.clear();

	for(map<string, TrainingData*>::iterator it = m_mTrain.begin(); it != m_mTrain.end(); it++)
	{
		delete it->second;
	}
	m_mTrain.clear();

	for(map<string, TrainingData*>::iterator it = m_mValidate.begin(); it != m_mValidate.end(); it++)
	{
		delete it->second;
	}
	m_mValidate.clear();

	for(map<string, umSVM*>::iterator it = m_mSVMs.begin(); it != m_mSVMs.end(); it++)
	{
		delete it->second;
	}
	m_mSVMs.clear();
}

void mai::CatalogueDetection::processPipeline()
{

	Size cellSize = Size(Constants::HOG_CELLSIZE, Constants::HOG_CELLSIZE);
	Size blockStride = Size(Constants::HOG_BLOCKSTRIDE, Constants::HOG_BLOCKSTRIDE);
	Size blockSize = Size(Constants::HOG_BLOCKSIZE, Constants::HOG_BLOCKSIZE);
	Size imageSize = Size(Constants::HOG_IMAGE_SIZE_X, Constants::HOG_IMAGE_SIZE_Y);

	// Not sure about these
	Size winStride = Size(0,0);
	Size padding = Size(0,0);

	int iNumBins = Constants::HOG_BINS;

	int iDataSetDivider = Constants::DATESET_DIVIDER;

	computeHOG(imageSize,
			blockSize,
			blockStride,
			cellSize,
			iNumBins,
			winStride,
			padding);

	trainSVMs(iDataSetDivider);

	if(iDataSetDivider > 1)
	{
		predict();
	}
}

void mai::CatalogueDetection::computeHOG(Size imageSize,
		Size blockSize,
		Size blockStride,
		Size cellSize,
		int iNumBins,
		Size winStride,
		Size padding)
{
	for(map<string, DataSet*>::iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		umHOG::computeHOGForDataSet(it->second,
					imageSize,
					blockSize,
					blockStride,
					cellSize,
					iNumBins,
					winStride,
					padding);

		if(Constants::WRITE_HOG_IMAGES)
		{
			string strName = it->first;
			string strPath = "out";

			IOUtils::writeHOGImages(it->second,
					strPath,
					strName,
					imageSize,
					cellSize,
					blockSize,
					blockStride,
					iNumBins,
					Constants::HOG_VIZ_SCALEFACTOR,
					Constants::HOG_VIZ_VIZFACTOR);
		}
	}

}

void mai::CatalogueDetection::trainSVMs(int iDataSetDivider,
		bool bSearchSupportVectors)
{
	map<string, vector<vector<float> > > mPositiveTrain;
	map<string, vector<vector<float> > > mPositiveValidate;

	divideDataSets(mPositiveTrain, mPositiveValidate, iDataSetDivider);

	if(mPositiveTrain.size() < 2)
	{
		cout << "[mai::CatalogueDetection::trainSVMs] ERROR! At least 2 categories needed." << endl;
		return;
	}

	map<string, vector<vector<float> > > mNegativeTrain;
	map<string, vector<vector<float> > > mNegativeValidate;

	collectRandomNegatives(mPositiveTrain, mNegativeTrain);
	collectRandomNegatives(mPositiveValidate, mNegativeValidate);

	setupTrainingData(m_mTrain, mPositiveTrain, mNegativeTrain);
	setupTrainingData(m_mValidate, mPositiveValidate, mNegativeValidate);

	for(map<string, TrainingData*>::const_iterator it = m_mTrain.begin(); it != m_mTrain.end(); it++)
	{
		string strName = it->first;
		TrainingData* data = it->second;
		umSVM* svm = new umSVM();
		vector<vector<float> > vSupport;

//		std::string strDataname = "trainingdata";
//		IOUtils::writeMatToCSV(data->getData(), strDataname);
//		std::string strLabelname = "labeldata";
//		IOUtils::writeMatToCSV(data->getLabels(), strLabelname);

		svm->trainSVM(data->getData(), data->getLabels(), vSupport);

		m_mSVMs.insert(pair<string, umSVM*>(strName, svm));

		svm->saveSVM(strName);

		if(bSearchSupportVectors)
		{
			cout << "[mai::CatalogueDetection::trainSVMs] Searching support vectors in positives .." << endl;

			umSVM::searchSupportVector(mPositiveTrain.at(strName), vSupport);

			cout << "[mai::CatalogueDetection::trainSVMs] Searching support vectors in negatives .." << endl;

			umSVM::searchSupportVector(mNegativeTrain.at(strName), vSupport);

			cout << "[mai::CatalogueDetection::trainSVMs] Searching support vectors done." << endl;
		}
	}
}

void mai::CatalogueDetection::predict()
{
	map<string, Mat> mResults;

	for(std::map<std::string, TrainingData*>::const_iterator it = m_mValidate.begin();
			it != m_mValidate.end(); it++)
	{
		string strName = it->first;
		umSVM* svm = m_mSVMs.at(strName);
		Mat results(it->second->getData().rows, 1, CV_32SC1);

		svm->predict(it->second->getData(), results);

		int iResultsRows = results.rows;
		int iCorrectDetection = 0;

		cout << "[mai::CatalogueDetection::predict] SVM prediction result for label " << strName << " has " << iResultsRows << " rows." << endl;

		for(int i = 0; i < iResultsRows; ++i)
		{
			int iResultLabel = results.at<float>(i);
			int iDataLabel = it->second->getLabels().at<int>(i);

			if(iResultLabel == iDataLabel)
				iCorrectDetection++;

			if(Constants::DEBUG_SVM_PREDICTION)
				cout << "[mai::CatalogueDetection::predict] SVM predict for image " << i << " is " << iResultLabel << ". Label was : " << iDataLabel << endl;
		}

		double dCorrectDetectionRatio = (double)iCorrectDetection / iResultsRows;
		cout << "[mai::CatalogueDetection::predict] Ratio of correct detections for label " << strName << " : " << dCorrectDetectionRatio << endl;

		mResults.insert(pair<string, Mat>(strName, results));
	}
}

void mai::CatalogueDetection::divideDataSets(map<string, vector<vector<float> > > &mTrain,
			map<string, vector<vector<float> > > &mValidate,
			int iDataSetDivider)
{
	for(map<string, DataSet*>::const_iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		vector<vector<float> > vTrain, vValidate;

		it->second->getDescriptorsSeparated(iDataSetDivider, vValidate, vTrain);

		mTrain.insert(pair<string, vector<vector<float> > >(it->first, vTrain));
		mValidate.insert(pair<string, vector<vector<float> > >(it->first, vValidate));
	}
}

void mai::CatalogueDetection::collectRandomNegatives(map<string, vector<vector<float> > > &mPositives,
		map<string, vector<vector<float> > > &mNegatives)
{
	for(map<string,vector<vector<float> > >::const_iterator itTrainCategory = mPositives.begin();
			itTrainCategory != mPositives.end(); itTrainCategory++)
	{
		int iPosSampleSize = itTrainCategory->second.size();
		int iSamplesPerCategory = iPosSampleSize / (mPositives.size() - 1);
		string strKey = itTrainCategory->first;

		for(map<string, vector<vector<float> > >::const_iterator itTrainOthers = mPositives.begin();
				itTrainOthers != mPositives.end(); itTrainOthers++)
		{
			if(itTrainCategory != itTrainOthers)
			{
				int iCurrentSampleSize = itTrainOthers->second.size();
				vector<int> vIndices;

				for(int i = 0; i < iCurrentSampleSize; ++i)
				{
					vIndices.push_back(i);
				}
				random_shuffle(vIndices.begin(), vIndices.end());

				vector<vector<float> > vNegative;

				for(int i = 0; i < iSamplesPerCategory; ++i)
				{
					vNegative.push_back(itTrainOthers->second[vIndices[i]]);
				}

				if(mNegatives.count(strKey) == 0)
				{
					mNegatives.insert(std::pair<std::string, std::vector<std::vector<float> > >(strKey, vNegative));
				}
				else
				{
					mNegatives.at(strKey).insert(std::end(mNegatives.at(strKey)), std::begin(vNegative), std::end(vNegative));
				}
			}
		}
	}
}

void mai::CatalogueDetection::setupTrainingData(map<string, TrainingData*> &mTrainingData,
			map<string, vector<vector<float> > > &mPositives,
			map<string, vector<vector<float> > > &mNegatives)
{
	for(map<string, DataSet*>::const_iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		string strKey = it->first;
		vector<vector<float> > vPositives, vNegatives;
		vPositives = mPositives.at(strKey);
		vNegatives = mNegatives.at(strKey);

		TrainingData* td = new TrainingData(vPositives, vNegatives);

		mTrainingData.insert(pair<string, TrainingData*>(strKey, td));
	}
}
