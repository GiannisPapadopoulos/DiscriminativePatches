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
#include "Configuration.h"
#include "svm/umSVM.h"
#include "data/DataSet.h"
#include "data/TrainingData.h"
#include "IO/IOUtils.h"
#include "featureExtraction/umHOG.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>


using namespace cv;

using namespace std;


mai::CatalogueDetection::CatalogueDetection(Configuration* config)
:	m_Config(config)
{
	IOUtils::loadCatalogue(m_mCatalogue, IMREAD_COLOR, m_Config->getDataFilepath(), true);

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

	delete m_Config;
}

void mai::CatalogueDetection::processPipeline()
{

	Size cellSize = m_Config->getCellSize();
	Size blockStride = m_Config->getBlockStride();
	Size blockSize = m_Config->getBlockSize();
	Size imageSize = m_Config->getImageSize();

	// Not sure about these
	Size winStride = Size(0,0);
	Size padding = Size(0,0);

	int iNumBins = m_Config->getNumBins();

	int iDataSetDivider = m_Config->getDataSetDivider();

	bool bPredictTrainingData = m_Config->getPredictTrainingData();

	bool bWriteHOGImages= m_Config->getWriteHogImages();

	bool bApplyPCA = m_Config->getApplyPCA();

	computeHOG(imageSize,
			blockSize,
			blockStride,
			cellSize,
			iNumBins,
			winStride,
			padding,
			bWriteHOGImages,
			bApplyPCA);

	if(trainSVMs(iDataSetDivider))
	{
		if(bPredictTrainingData)
		{
			cout << "#-------------------------------------------------------------------------------#" << endl;
			cout << "[mai::CatalogueDetection::processPipeline] SVM prediction on training data." << endl;

			map<string, Mat> mTrainingResults;
			predict(m_mTrain, mTrainingResults);

			cout << "#-------------------------------------------------------------------------------#" << endl;
		}

		if(iDataSetDivider > 1)
		{
			cout << "#-------------------------------------------------------------------------------#" << endl;
			cout << "[mai::CatalogueDetection::processPipeline] SVM prediction on validation data." << endl;

			map<string, Mat> mValidationResults;
			predict(m_mValidate, mValidationResults);

			cout << "#-------------------------------------------------------------------------------#" << endl;
		}
	}
}

void mai::CatalogueDetection::computeHOG(Size imageSize,
		Size blockSize,
		Size blockStride,
		Size cellSize,
		int iNumBins,
		Size winStride,
		Size padding,
		bool bWriteHOGImages,
		bool bApplyPCA)
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
					padding,
					bApplyPCA);

		if(bWriteHOGImages)
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
					m_Config->getHogVizImageScalefactor(),
					m_Config->getHogVizBinScalefactor());
		}
	}

}

bool mai::CatalogueDetection::trainSVMs(int iDataSetDivider,
		bool bSearchSupportVectors)
{
	// Positive training and validation data per named category
	map<string, vector<vector<float> > > mPositiveTrain;
	map<string, vector<vector<float> > > mPositiveValidate;

	divideDataSets(mPositiveTrain, mPositiveValidate, iDataSetDivider);

	if(mPositiveTrain.size() < 2)
	{
		cout << "[mai::CatalogueDetection::trainSVMs] ERROR! At least 2 categories needed." << endl;
		return false;
	}

	// Negative training and validation data per named category
	map<string, vector<vector<float> > > mNegativeTrain;
	map<string, vector<vector<float> > > mNegativeValidate;

	collectRandomNegatives(mPositiveTrain, mNegativeTrain);
	collectRandomNegatives(mPositiveValidate, mNegativeValidate);

	setupTrainingData(m_mTrain, mPositiveTrain, mNegativeTrain);
	setupTrainingData(m_mValidate, mPositiveValidate, mNegativeValidate);

	// Train svms for each category
	for(map<string, TrainingData*>::const_iterator it = m_mTrain.begin(); it != m_mTrain.end(); it++)
	{
		string strName = it->first;
		TrainingData* data = it->second;
		umSVM* svm = new umSVM(m_Config->getSvmCValue());
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

	return true;
}

void mai::CatalogueDetection::predict(map<string, TrainingData*> &mData,
		map<string, Mat> &mResults)
{
	// Predict data by trained svms for each category
	for(map<string, TrainingData*>::const_iterator it = mData.begin();
			it != mData.end(); it++)
	{
		string strName = it->first;
		umSVM* svm = m_mSVMs.at(strName);

		// result matrix containing predicted labels
		Mat results(it->second->getData().rows, 1, CV_32SC1);

		svm->predict(it->second->getData(), results);

		int iResultsRows = results.rows;
		int iCorrectDetection = 0;

		cout << "[mai::CatalogueDetection::predict] SVM prediction result for label " << strName << " has " << iResultsRows << " rows." << endl;

		// Compare predicted labels with original ones
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
	// Divide datasets of each catefory into exclusive training and validation parts
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
		// Find number of desired negative features according to positive ones
		int iPosSampleSize = itTrainCategory->second.size();
		int iSamplesPerCategory = iPosSampleSize / (mPositives.size() - 1);
		string strKey = itTrainCategory->first;

		// Collect negative samples from all other categories
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
	// Setup training data for each category
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
