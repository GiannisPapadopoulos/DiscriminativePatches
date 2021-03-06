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

#include "CatalogTraining.h"

#include "CatalogClassificationSVM.h"
#include "../data/DataSet.h"
#include "../data/TrainingData.h"
#include "../IO/IOUtils.h"
#include "../featureExtraction/umHOG.h"
#include "../utils/FaceDetection.h"
#include "../configuration/Configuration.h"
#include "../configuration/Constants.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>



using namespace cv;
using namespace std;

mai::CatalogTraining::CatalogTraining(const Configuration* const config)
:	m_Config(config)
,	m_Classifiers(new CatalogClassificationSVM(config))
{}

mai::CatalogTraining::~CatalogTraining()
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

	delete m_Classifiers;
}

void mai::CatalogTraining::processPipeline()
{
	cout << "[mai::CatalogueDetection::processPipeline] Loading data ..." << endl;

	if(!IOUtils::loadCatalogue(m_mCatalogue, IMREAD_GRAYSCALE, m_Config->getDataFilepath(), m_Config->getAddFlipedImages()))
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

	if(m_Config->getApplicationMode() == Configuration::appMode::Retrain)
	{
		cout << "[mai::CatalogueDetection::processPipeline] Loading svms for retraining ..." << endl;
		string strSVMInputPath = m_Config->getSVMInputPath();

		if(!m_Classifiers->loadSVMs(strSVMInputPath))
		{
			cout << "[mai::CatalogueDetection::processPipeline] WARNING! Loading svms for retraining from " << strSVMInputPath
					<< " failed. Training new ones." << endl;
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

	bool bCrossValidate = m_Config->getCrossValidate();

	map<string, Mat> mTrainingResults;
	map<string, Mat> mValidationResults;

	computeHOG(imageSize,
			blockSize,
			blockStride,
			cellSize,
			iNumBins,
			winStride,
			padding,
			bWriteHOGImages,
			bApplyPCA);

	if(m_Config->getPerfromClustering())
	{
		cout << "[mai::CatalogueDetection::processPipeline] clustering data ..." << endl;

		performClustering(imageSize,
			blockSize,
			blockStride,
			cellSize,
			iNumBins);
	}

	cout << "[mai::CatalogueDetection::processPipeline] Setting up svm data ..." << endl;
	setupSVMData(iDataSetDivider);

	cout << "[mai::CatalogueDetection::processPipeline] Training svm ..." << endl;

	m_Classifiers->trainSVMs(m_mTrain, m_Config->getSvmCValue());

	cout << "[mai::CatalogueDetection::processPipeline] Training SVM done." << endl;

	if(bPredictTrainingData && !bCrossValidate)
	{
		cout << "#-------------------------------------------------------------------------------#" << endl;
		cout << "[mai::CatalogueDetection::processPipeline] SVM prediction on training data." << endl;

		m_Classifiers->predict(m_mTrain, mTrainingResults);

		cout << "#-------------------------------------------------------------------------------#" << endl;
	}

	if(iDataSetDivider > 1)
	{
		cout << "#-------------------------------------------------------------------------------#" << endl;
		cout << "[mai::CatalogueDetection::processPipeline] SVM prediction on validation data." << endl;

		m_Classifiers->predict(m_mValidate, mValidationResults);

		cout << "#-------------------------------------------------------------------------------#" << endl;

		if(bCrossValidate)
		{
			cout << "[mai::CatalogueDetection::processPipeline] Training SVM with validation data for cross validation ..." << endl;

			m_Classifiers->trainSVMs(m_mValidate, m_Config->getSvmCValue());

			cout << "#-------------------------------------------------------------------------------#" << endl;
			cout << "[mai::CatalogueDetection::processPipeline] SVM prediction on training data swapped for validation." << endl;

			m_Classifiers->predict(m_mTrain, mTrainingResults);

			cout << "#-------------------------------------------------------------------------------#" << endl;
		}
	}

	if(m_Config->getWriteSvMs())
	{
		string strSVMPath = m_Config->getSvmOutputPath();
		cout << "[mai::CatalogueDetection::processPipeline] Saving SVM to " << strSVMPath << endl;

		IOUtils::writeSVMs(m_Classifiers->getSVMs(), strSVMPath);
		cout << "[mai::CatalogueDetection::processPipeline] Saving SVM done." << endl;
	}
}

void mai::CatalogTraining::detectFaces()
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
			// replace original image with face part
			delete it->second;
			it->second = faces;
		}
		else
		{
			cout << "[mai::CatalogueDetection::detectFaces] ERROR! No faces found for images in " << it->first << endl;
		}
	}
}

void mai::CatalogTraining::computeHOG(const Size imageSize,
		const Size blockSize,
		const Size blockStride,
		const Size cellSize,
		const int iNumBins,
		const Size winStride,
		const Size padding,
		const bool bWriteHOGImages,
		const bool bApplyPCA)
{
	for(map<string, DataSet*>::iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		DataSet* pDataset = it->second;

		umHOG::computeHOGForDataSet(pDataset,
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
			string strPath = m_Config->getHogOutputPath();

			IOUtils::writeHOGImages(pDataset,
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

		// free memory
		pDataset->removeImages();
	}
}

void mai::CatalogTraining::performClustering(const Size imageSize,
		const Size blockSize,
		const Size blockStride,
		const Size cellSize,
		const int iNumBins)
{
	int cellsPerBlock = (blockSize.width / cellSize.width) * (blockSize.height / cellSize.height);
	int blockFeatures = cellsPerBlock * iNumBins;
	int blocksPerRow = imageSize.width / blockStride.width - (blockSize.width / blockStride.width - 1) ;
	int blocksPerColumn = imageSize.height / blockStride.height - (blockSize.height / blockStride.height - 1) ;

	for(map<string, DataSet*>::iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		DataSet* pDataset = it->second;

		// Extract features from all images in dataset.
		for(unsigned int i = 0; i < pDataset->getImageCount(); ++i)
		{
			vector<float> descriptorsValues;
			pDataset->getDescriptorValuesFromImageAt(i, descriptorsValues);

			vector<vector<vector<float>>> patchDescriptorValues(blocksPerColumn);
			for(int i = 0; i < blocksPerColumn; i++) {
				patchDescriptorValues[i] = vector<vector<float>>(blocksPerRow);
			}

			for(int i = 0; i < blocksPerColumn; i++) {
				for(int j = 0; j < blocksPerRow; j++) {
					patchDescriptorValues[i][j] = vector<float>(blockFeatures);
					int block = blocksPerColumn * i + j;
					std::copy(descriptorsValues.begin() + block * blockFeatures, descriptorsValues.begin() + (block +1) * blockFeatures, patchDescriptorValues[i][j].begin());
				}
			}
			pDataset->addPatchDescriptorValuesToImageAt(i, patchDescriptorValues);

			//		cout << patchDescriptorValues.size() << " " << patchDescriptorValues[0].size() << " " << patchDescriptorValues[0][0].size() << endl;
		}
	}
}

void mai::CatalogTraining::setupSVMData(const int iDataSetDivider)
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

void mai::CatalogTraining::divideDataSets(map<string, vector<vector<float> > > &mTrain,
			map<string, vector<vector<float> > > &mValidate,
			const int iDataSetDivider)
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

void mai::CatalogTraining::collectRandomNegatives(const map<string, vector<vector<float> > > &mPositives,
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
		mSampleSizes.clear();
	}
}

void mai::CatalogTraining::calculateSampleSizes(const string &strKey,
		const map<string, int> &mFeatureSizes,
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
	do
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
	while(iRest > 0);

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

void mai::CatalogTraining::setupTrainingData(map<string, TrainingData*> &mTrainingData,
			const map<string, vector<vector<float> > > &mPositives,
			const map<string, vector<vector<float> > > &mNegatives)
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
