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
#include "featureExtraction/cvHOG.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>


using namespace cv;

using namespace std;


mai::CatalogueDetection::CatalogueDetection()
{}

mai::CatalogueDetection::~CatalogueDetection()
{
	for(std::map<std::string, DataSet*>::iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); it++)
	{
		delete it->second;
	}
	m_mCatalogue.clear();

	for(std::map<std::string, TrainingData*>::iterator it = m_mTrain.begin(); it != m_mTrain.end(); it++)
	{
		delete it->second;
	}
	m_mTrain.clear();

	for(std::map<std::string, TrainingData*>::iterator it = m_mValidate.begin(); it != m_mValidate.end(); it++)
	{
		delete it->second;
	}
	m_mValidate.clear();
}

void mai::CatalogueDetection::catalogueDetection(std::string &strFilePath)
{
	IOUtils::loadCatalogue(m_mCatalogue, IMREAD_COLOR, strFilePath, true);

	cout << "[mai::UDoMLDP::CatalogueDetecion] Number of labels: " << m_mCatalogue.size() << endl;
	for (std::map<std::string, DataSet*>::const_iterator it = m_mCatalogue.begin(); it != m_mCatalogue.end(); ++it)
	{
		cout << "[mai::UDoMLDP::CatalogueDetecion] Label: " << it->first << ", number of images: " << it->second->getImageCount() << endl;
	}

	Size cellSize = Size(Constants::HOG_CELLSIZE, Constants::HOG_CELLSIZE);
	Size blockStride = Size(Constants::HOG_BLOCKSTRIDE, Constants::HOG_BLOCKSTRIDE);
	Size blockSize = Size(Constants::HOG_BLOCKSIZE, Constants::HOG_BLOCKSIZE);
	Size imageSize = Size(Constants::HOG_IMAGE_SIZE_X, Constants::HOG_IMAGE_SIZE_Y);

	// Not sure about these
	Size winStride = Size(0,0);
	Size padding = Size(0,0);


	this->computeHOGForCatalogue(m_mCatalogue,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			winStride,
			padding);

	this->trainSVMsForCatalogue(m_mCatalogue);
}

void mai::CatalogueDetection::computeHOGForCatalogue(std::map<std::string, DataSet*> &mCatalogue,
		Size imageSize,
		Size blockSize,
		Size blockStride,
		Size cellSize,
		Size winStride,
		Size padding)
{
	for(std::map<std::string, DataSet*>::iterator it = mCatalogue.begin(); it != mCatalogue.end(); it++)
	{
		this->computeHOGForDataSet(it->second,
					imageSize,
					blockSize,
					blockStride,
					cellSize,
					winStride,
					padding);

		if(Constants::WRITE_HOG_IMAGES)
		{
			std::string strName = it->first;
			std::string strPath = "out";

			IOUtils::writeHOGImages(it->second,
					strPath,
					strName,
					imageSize,
					cellSize,
					Constants::HOG_VIZ_SCALEFACTOR,
					Constants::HOG_VIZ_VIZFACTOR);
		}
	}

}

void mai::CatalogueDetection::computeHOGForDataSet(DataSet* data,
		Size imageSize,
		Size blockSize,
		Size blockStride,
		Size cellSize,
		Size winStride,
		Size padding)
{
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		const Mat* image = data->getImageAt(i);

		if(Constants::DEBUG_MAIN_ALG) {
		  cout << "[mai::UDoMLDP::computeHOGForDataSet] resizing image to " << imageSize << endl;
		}
		Mat resizedImage;
		cv::resize(*image, resizedImage, imageSize);

		vector< float> descriptorsValues;

		cvHOG::extractFeatures(descriptorsValues, resizedImage, blockSize, blockStride, cellSize, Constants::HOG_BINS, winStride, padding);

		if(Constants::DEBUG_MAIN_ALG) {
			cout << "[mai::UDoMLDP::computeHOGForDataSet] Number of descriptors: " << descriptorsValues.size() << endl;
		}

		data->addDescriptorValuesToImageAt(i, descriptorsValues);
	}
}

void mai::CatalogueDetection::trainSVMsForCatalogue(std::map<std::string, DataSet*> &mCatalogue)
{
	std::map<std::string, std::vector<std::vector<float> > > mPositiveTrain;
	std::map<std::string, std::vector<std::vector<float> > > mPositiveValidate;

	this->collectPositivesFromCatalogue(mCatalogue, mPositiveTrain, mPositiveValidate);

	if(mPositiveTrain.size() < 2)
	{
		std::cout << "[mai::UDoMLDP::trainSVMsForCatalogue] ERROR! At least 2 categories needed." << std::endl;
		return;
	}

	std::map<std::string, std::vector<std::vector<float> > > mNegativeTrain;
	std::map<std::string, std::vector<std::vector<float> > > mNegativeValidate;

	this->collectRandomNegatives(mPositiveTrain, mNegativeTrain);
	this->collectRandomNegatives(mPositiveValidate, mNegativeValidate);

	this->setupTrainingData(m_mTrain, m_mCatalogue, mPositiveTrain, mNegativeTrain);
	this->setupTrainingData(m_mValidate, m_mCatalogue, mPositiveValidate, mNegativeValidate);

	for(std::map<std::string, TrainingData*>::const_iterator it = m_mTrain.begin(); it != m_mTrain.end(); it++)
	{
		std::string strName = it->first;
		TrainingData* data = it->second;
		umSVM* svm = new umSVM();
		std::vector<std::vector<float> > vSupport;

//		std::string strDataname = "trainingdata";
//		IOUtils::writeMatToCSV(data->getData(), strDataname);
//		std::string strLabelname = "labeldata";
//		IOUtils::writeMatToCSV(data->getLabels(), strLabelname);

		svm->trainSVM(data->getData(), data->getLabels(), vSupport);

		m_mSVMs.insert(std::pair<std::string, umSVM*>(strName, svm));

		svm->saveSVM(strName);
	}


	//	cout << "Searching support vectors in positives .." << endl;
	//
	//	searchSupportVector(positives, vSupport);
	//
	//	cout << "Searching support vectors in negatives .." << endl;
	//
	//	searchSupportVector(negatives, vSupport);
	//
	//	cout << "Searching support vectors done." << endl;
//		std::vector<Mat*> images = it->second->
//		int iPercentageValidationImages = images.size()/Constants::DATESET_DIVIDER > 1 ? images.size()/Constants::DATESET_DIVIDER : 1;
//
//		std::vector<Mat*> images2HalfNeg(std::make_move_iterator(images.begin() + iPercentageValidationImages), std::make_move_iterator(images.end()));
//		images.erase(images.begin() + iPercentageValidationImages, images.end());
//
//		m_pNegativeTrain->setImages(images2HalfNeg);
//		m_pNegativeValid->setImages(images);


}

void mai::CatalogueDetection::collectPositivesFromCatalogue(std::map<std::string, DataSet*> &mCatalogue,
			std::map<std::string, std::vector<std::vector<float> > > &mTrain,
			std::map<std::string, std::vector<std::vector<float> > > &mValidate)
{
	for(std::map<std::string, DataSet*>::const_iterator it = mCatalogue.begin(); it != mCatalogue.end(); it++)
	{
		std::vector<std::vector<float> > vTrain, vValidate;

		it->second->getDescriptorsSeparated(Constants::DATESET_DIVIDER, vValidate, vTrain);

		mTrain.insert(std::pair<std::string, std::vector<std::vector<float> > >(it->first, vTrain));
		mValidate.insert(std::pair<std::string, std::vector<std::vector<float> > >(it->first, vValidate));
	}
}

void mai::CatalogueDetection::collectRandomNegatives(std::map<std::string, std::vector<std::vector<float> > > &mPositives,
		std::map<std::string, std::vector<std::vector<float> > > &mNegatives)
{
	for(std::map<std::string, std::vector<std::vector<float> > >::const_iterator itTrainCategory = mPositives.begin();
			itTrainCategory != mPositives.end(); itTrainCategory++)
	{
		int iPosSampleSize = itTrainCategory->second.size();
		int iSamplesPerCategory = iPosSampleSize / (mPositives.size() - 1);
		string strKey = itTrainCategory->first;

		std::cout << "sampling " << itTrainCategory->first << ", size " << iPosSampleSize << ", per category " << iSamplesPerCategory << std::endl;

		for(std::map<std::string, std::vector<std::vector<float> > >::const_iterator itTrainOthers = mPositives.begin();
				itTrainOthers != mPositives.end(); itTrainOthers++)
		{
			if(itTrainCategory != itTrainOthers)
			{
				int iCurrentSampleSize = itTrainOthers->second.size();
				std::vector<int> vIndices;

				for(int i = 0; i < iCurrentSampleSize; ++i)
				{
					vIndices.push_back(i);
				}
				std::random_shuffle(vIndices.begin(), vIndices.end());

				std::vector<std::vector<float> > vNegative;

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

void mai::CatalogueDetection::setupTrainingData(std::map<std::string, TrainingData*> &mTrainingData,
			std::map<std::string, DataSet*> &mCatalogue,
			std::map<std::string, std::vector<std::vector<float> > > &mPositives,
			std::map<std::string, std::vector<std::vector<float> > > &mNegatives)
{
	for(std::map<std::string, DataSet*>::const_iterator it = mCatalogue.begin(); it != mCatalogue.end(); it++)
	{
		string strKey = it->first;
		std::vector<std::vector<float> > vPositives, vNegatives;
		vPositives = mPositives.at(strKey);
		vNegatives = mNegatives.at(strKey);

		TrainingData* td = new TrainingData(vPositives, vNegatives);

		mTrainingData.insert(std::pair<std::string, TrainingData*>(strKey, td));
	}
}
