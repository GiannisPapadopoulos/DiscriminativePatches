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
#include "utils/FaceDetection.h"

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
{}

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
	cout << "[mai::CatalogueDetection::processPipeline] Loading data ..." << endl;

	if(!IOUtils::loadCatalogue(m_mCatalogue, IMREAD_GRAYSCALE, m_Config->getDataFilepath(), true))
		return;

	if(m_mCatalogue.size() < 2)
	{
		cout << "[mai::CatalogueDetection::processPipeline] ERROR! At least 2 categories are needed." << endl;
		return;
	}

	if(Constants::DEBUG_MAIN_ALG)
	{
		cout << "[mai::CatalogueDetection::processPipeline] Number of labels: " << m_mCatalogue.size() << endl;
		for (map<string, DataSet*>::const_iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); ++it)
		{
			cout << "[mai::CatalogueDetection::processPipeline] Label: " << it->first << ", number of images: " << it->second->getImageCount() << endl;
		}
	}

	if(m_Config->getDetectFaces())
	{
		cout << "[mai::CatalogueDetection::processPipeline] Performing face detection ..." << endl;
		detectFaces();
	}

	cout << "[mai::CatalogueDetection::processPipeline] Computing HOG descriptors ..." << endl;

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

	cout << "[mai::CatalogueDetection::processPipeline] Setting up svm data ..." << endl;
	setupSVMData(iDataSetDivider);

	cout << "[mai::CatalogueDetection::processPipeline] Training svm ..." << endl;
	trainAndSaveSVMs();

	cout << "[mai::CatalogueDetection::processPipeline] Training SVM done." << endl;

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

void mai::CatalogueDetection::detectFaces()
{
	for(map<string, DataSet*>::iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		string strFilename = m_Config->getCascadeFilterFileName();
		double dScale = m_Config->getFDScale();
		int iMinNeighbors = m_Config->getFDMinNeighbors();
		Size minSize = m_Config->getFDMinSize();
		Size maxSize = m_Config->getFDMaxSize();

		DataSet* faces = FaceDetection::detectFaces(it->second,
				strFilename,
				dScale,
				iMinNeighbors,
				minSize,
				maxSize);

		if(faces != NULL)
		{
			delete it->second;
			it->second = faces;
		}
		else
		{
			cout << "[mai::CatalogueDetection::detectFaces] ERROR! No faces found for images in " << it->first << endl;
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
			string strPath = "outHOG";

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

void mai::CatalogueDetection::setupSVMData(int iDataSetDivider)
{
	// Positive training and validation data per named category
	map<string, vector<vector<float> > > mPositiveTrain;
	map<string, vector<vector<float> > > mPositiveValidate;

	divideDataSets(mPositiveTrain, mPositiveValidate, iDataSetDivider);

	if(Constants::DEBUG_MAIN_ALG)
	{
		cout << "[mai::CatalogueDetection::setupSVMData] Dataset division done. Collecting data ..." << endl;
	}

	// Negative training and validation data per named category
	map<string, vector<vector<float> > > mNegativeTrain;
	map<string, vector<vector<float> > > mNegativeValidate;

	if(Constants::DEBUG_MAIN_ALG)
	{
		cout << "[mai::CatalogueDetection::setupSVMData] Collecting random negatives for training data." << endl;
	}
	collectRandomNegatives(mPositiveTrain, mNegativeTrain);

	if(Constants::DEBUG_MAIN_ALG)
	{
		cout << "[mai::CatalogueDetection::setupSVMData] Collecting random negatives for validation data." << endl;
	}
	collectRandomNegatives(mPositiveValidate, mNegativeValidate);

	if(Constants::DEBUG_MAIN_ALG)
	{
		cout << "[mai::CatalogueDetection::setupSVMData] Data collection done. Setting up training data ..." << endl;
	}

	setupTrainingData(m_mTrain, mPositiveTrain, mNegativeTrain);
	setupTrainingData(m_mValidate, mPositiveValidate, mNegativeValidate);
}

void mai::CatalogueDetection::trainAndSaveSVMs()
{
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

		if(Constants::DEBUG_MAIN_ALG)
		{
			cout << "[mai::CatalogueDetection::trainSVMs] svm traind for " << strName << endl;
		}
	}
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
	// Count all sample vector sizes
	map<string, int> mFeatureSizes;

	for(map<string, vector<vector<float> > >::const_iterator itCategory = mPositives.begin();
			itCategory != mPositives.end(); itCategory++)
	{
		string strKey = itCategory->first;
		int iSampleSize = itCategory->second.size();
		mFeatureSizes.insert(pair<string, int>(strKey, iSampleSize));
	}

	// Collect data from categories
	for(map<string, vector<vector<float> > >::const_iterator itTrainCategory = mPositives.begin();
			itTrainCategory != mPositives.end(); itTrainCategory++)
	{
		string strKey = itTrainCategory->first;

		map<string, int> mSampleSizes;
		calculateSampleSizes(strKey, mFeatureSizes, mSampleSizes);

		// Collect negative samples from all other categories
		for(map<string, vector<vector<float> > >::const_iterator itTrainOthers = mPositives.begin();
				itTrainOthers != mPositives.end(); itTrainOthers++)
		{
			if(itTrainCategory != itTrainOthers)
			{
				int iCurrentSampleSize = mSampleSizes.at(itTrainOthers->first);
						//itTrainOthers->second.size();
				vector<int> vIndices;

				for(int i = 0; i < iCurrentSampleSize; ++i)
				{
					vIndices.push_back(i);
				}
				random_shuffle(vIndices.begin(), vIndices.end());

				vector<vector<float> > vNegative;

				//  && i < mSampleSizes.at(itTrainOthers->first)
				for(int i = 0; i < iCurrentSampleSize; ++i)
				{
					vNegative.push_back(itTrainOthers->second[vIndices[i]]);
				}

				if(mNegatives.count(strKey) == 0)
				{
					mNegatives.insert(pair<string, vector<std::vector<float> > >(strKey, vNegative));
				}
				else
				{
					mNegatives.at(strKey).insert(end(mNegatives.at(strKey)), begin(vNegative), end(vNegative));
				}
			}
		}
	}
}

//iNumCategories = mPositives.size()
void mai::CatalogueDetection::calculateSampleSizes(string strKey,
		map<string, int> &mFeatureSizes,
		map<string, int> &mSampleSizes)
{
	// Get all feature sizes
	int iAllFeatureSizes = 0;
	for(map<string, int>::const_iterator itHelperCategory = mFeatureSizes.begin();
			itHelperCategory != mFeatureSizes.end(); itHelperCategory++)
	{
		string strHelperKey = itHelperCategory->first;
		if(strHelperKey.compare(strKey) != 0)
		{
			iAllFeatureSizes += itHelperCategory->second;
		}
	}

	// Find number of desired negative features according to positive ones
	int iPosSampleSize = mFeatureSizes.at(strKey);
	iPosSampleSize = iPosSampleSize < iAllFeatureSizes ? iPosSampleSize : iAllFeatureSizes;
	int iSamplesPerCategory = iPosSampleSize / (mFeatureSizes.size() - 1);
	int iRest = iPosSampleSize % (mFeatureSizes.size() - 1);

	// Helper for actual sample sizes
	mSampleSizes.clear();
	for(map<string, int>::const_iterator itHelperCategory = mFeatureSizes.begin();
			itHelperCategory != mFeatureSizes.end(); itHelperCategory++)
	{
		mSampleSizes.insert(pair<string, int>(itHelperCategory->first, iSamplesPerCategory));
	}

	// Disperse possible rest evenly
	while(iRest > 0)
	{
		map<string, int>::iterator itSampleSize = mSampleSizes.begin();
		map<string, int>::const_iterator itFeatureSize = mFeatureSizes.begin();
		while(itFeatureSize != mFeatureSizes.end()
				&& itSampleSize != mSampleSizes.end())
		{
			string strFeatureKey = itFeatureSize->first;
			if(strFeatureKey.compare(strKey) != 0)
			{
				int iFeatureSize = itFeatureSize->second;
				int iSampleSize = itSampleSize->second;
				if(iSampleSize > iFeatureSize)
				{
					iRest += iSampleSize - iFeatureSize;

					itSampleSize->second = iFeatureSize;
				}
				else if(iSampleSize + 1 <= iFeatureSize)
				{
					if(iRest > 0)
					{
						itSampleSize->second++;
						iRest--;
					}
				}
			}

			itFeatureSize++;
			itSampleSize++;
		}
	}

	if(Constants::DEBUG_MAIN_ALG)
	{
		cout << "[mai::CatalogueDetection::calculateSampleSizes] Collecting  " << iPosSampleSize << " negatives for " << strKey << endl;
		for(map<string, int>::const_iterator itSampleCategory = mSampleSizes.begin();
				itSampleCategory != mSampleSizes.end(); itSampleCategory++)
		{
			string strSampleKey = itSampleCategory->first;
			if(strSampleKey.compare(strKey) != 0)
			{
				cout << "[mai::CatalogueDetection::calculateSampleSizes] " << strSampleKey << " : " << itSampleCategory->second << endl;
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
