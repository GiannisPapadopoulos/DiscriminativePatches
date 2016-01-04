/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * UDoMLDP.cpp
 *
 *  Created on: Oct 25, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "UDoMLDP.h"
#include "Constants.h"

#include "svm/umSVM.h"
#include "data/DataSet.h"
#include "data/TrainingData.h"
#include "IO/IOUtils.h"
#include "featureExtraction/umPCA.h"
#include "utils/ImageDisplayUtils.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include "featureExtraction/umHOG.h"


using namespace cv;

using namespace std;


mai::UDoMLDP::UDoMLDP()
	: m_pPositiveTrain(new DataSet())
	, m_pNegativeTrain(new DataSet())
	, m_pPositiveValid(new DataSet())
	, m_pNegativeValid(new DataSet())
	, m_pSVM(new umSVM())
{}

mai::UDoMLDP::~UDoMLDP()
{
	delete m_pPositiveTrain;
	delete m_pNegativeTrain;
	delete m_pPositiveValid;
	delete m_pNegativeValid;

	delete m_pSVM;
}

void mai::UDoMLDP::basicDetecion(std::string &strFilePathPositives, std::string &strFilePathNegatives)
{
	std::vector<Mat*> images;
	std::vector<std::string> imageNames;

	IOUtils::loadImages ( images, IMREAD_COLOR, strFilePathPositives );
	cout << "Num positive images " << images.size() << endl;
	int iPercentageValidationImages = images.size()/Constants::DATESET_DIVIDER > 1 ? images.size()/Constants::DATESET_DIVIDER : 1;

	std::string p = "equal";
	std::string f = "p";
	IOUtils::writeImages(images, p, f);

	std::vector<Mat*> images2Half(std::make_move_iterator(images.begin() + iPercentageValidationImages), std::make_move_iterator(images.end()));
	images.erase(images.begin() + iPercentageValidationImages, images.end());

	IOUtils::addFlippedImages( images2Half, imageNames, 1 );
	IOUtils::addFlippedImages( images, imageNames, 1 );

	m_pPositiveTrain->setImages(images2Half, imageNames);
	m_pPositiveValid->setImages(images, imageNames);

	int iMH, iMW;
	m_pPositiveTrain->getMaxDImensions(iMW, iMH);
	cout << "Num positive training images: " << m_pPositiveTrain->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;
	m_pPositiveValid->getMaxDImensions(iMW, iMH);
	cout << "Num positive validation images: " << m_pPositiveValid->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;

	images.clear();

	IOUtils::loadImages ( images, IMREAD_COLOR, strFilePathNegatives );
	cout << "Num negative images " << images.size() << endl;
	iPercentageValidationImages = images.size()/Constants::DATESET_DIVIDER > 1 ? images.size()/Constants::DATESET_DIVIDER : 1;

	std::vector<Mat*> images2HalfNeg(std::make_move_iterator(images.begin() + iPercentageValidationImages), std::make_move_iterator(images.end()));
	images.erase(images.begin() + iPercentageValidationImages, images.end());

	IOUtils::addFlippedImages( images2HalfNeg, imageNames, 1 );
	IOUtils::addFlippedImages( images, imageNames, 1 );

	m_pNegativeTrain->setImages(images2HalfNeg, imageNames);
	m_pNegativeValid->setImages(images, imageNames);

	m_pNegativeTrain->getMaxDImensions(iMW, iMH);
	cout << "Num negative training images: " << m_pNegativeTrain->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;
	m_pNegativeValid->getMaxDImensions(iMW, iMH);
	cout << "Num negative validation images: " << m_pNegativeValid->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;

	Size cellSize = Size(Constants::HOG_CELLSIZE, Constants::HOG_CELLSIZE);
	Size blockStride = Size(Constants::HOG_BLOCKSTRIDE, Constants::HOG_BLOCKSTRIDE);
	Size blockSize = Size(Constants::HOG_BLOCKSIZE, Constants::HOG_BLOCKSIZE);
	Size imageSize = Size(Constants::HOG_IMAGE_SIZE_X, Constants::HOG_IMAGE_SIZE_Y);

	// Not sure about these
	Size winStride = Size(0,0);
	Size padding = Size(0,0);


	umHOG::computeHOGForDataSet(m_pPositiveTrain,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			Constants::HOG_BINS,
			winStride,
			padding);
	umHOG::computeHOGForDataSet(m_pNegativeTrain,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			Constants::HOG_BINS,
			winStride,
			padding);
	umHOG::computeHOGForDataSet(m_pPositiveValid,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			Constants::HOG_BINS,
			winStride,
			padding);
	umHOG::computeHOGForDataSet(m_pNegativeValid,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			Constants::HOG_BINS,
			winStride,
			padding);

	if(Constants::WRITE_HOG_IMAGES) {

	  std::string strName = "positiveTrain";
	  std::string strPath = "out";

    IOUtils::writeHOGImages(m_pPositiveTrain,
        strPath,
        strName,
        imageSize,
        cellSize,
		blockSize,
		blockStride,
		Constants::HOG_BINS,
        Constants::HOG_VIZ_SCALEFACTOR,
        Constants::HOG_VIZ_VIZFACTOR);

    strName = "positiveValid";
    IOUtils::writeHOGImages(m_pPositiveValid,
          strPath,
          strName,
          imageSize,
          cellSize,
		  blockSize,
		  blockStride,
		  Constants::HOG_BINS,
          Constants::HOG_VIZ_SCALEFACTOR,
          Constants::HOG_VIZ_VIZFACTOR);

    strName = "negativeTrain";
    IOUtils::writeHOGImages(m_pNegativeTrain,
          strPath,
          strName,
          imageSize,
          cellSize,
		  blockSize,
		  blockStride,
		  Constants::HOG_BINS,
          Constants::HOG_VIZ_SCALEFACTOR,
          Constants::HOG_VIZ_VIZFACTOR);

    strName = "negativeValid";
    IOUtils::writeHOGImages(m_pNegativeValid,
          strPath,
          strName,
          imageSize,
          cellSize,
		  blockSize,
		  blockStride,
		  Constants::HOG_BINS,
          Constants::HOG_VIZ_SCALEFACTOR,
          Constants::HOG_VIZ_VIZFACTOR);
	}

	trainSVMOnDataSets(m_pPositiveTrain, m_pNegativeTrain);

	// Prediction has to correspond with training data !!
	// check code of trainSVMOnDataSets

	//predictDataSetbySVM(m_pPositiveValid);

	//predictDataSetbySVM(m_pNegativeValid);

	cout << "Positives prediction" << endl;
	//predictWholeDataSetbySVMForSinglePatchImage(m_pPositiveValid);

	int numTruePositives = predictDataSetbySVMForSinglePatchImage(m_pPositiveTrain);

	cout << "correctly predicted positives: " << numTruePositives << " out of " << m_pPositiveTrain->getImageCount() << endl;

	int numFalsePositives =	predictDataSetbySVMForSinglePatchImage(m_pNegativeTrain);
	int correctlyPredictedNegatives = m_pNegativeTrain->getImageCount() - numFalsePositives;

	cout << "correctly predicted negatives: " << correctlyPredictedNegatives << " out of " << m_pNegativeTrain->getImageCount() << endl;
}

void mai::UDoMLDP::predictDataSetbySVM(DataSet* data)
{
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		vector<float> descriptorsValues;
		data->getDescriptorValuesFromImageAt(i, descriptorsValues);

		for(unsigned int j = 0; j < descriptorsValues.size(); ++j)
		{
			Mat sampleMat = (Mat_<float>(1,1) << descriptorsValues[j]);

			float fResultLabel = m_pSVM->predict(sampleMat, false);
			float fResultValue = m_pSVM->predict(sampleMat, true);

			cout << "SVM predict for " << descriptorsValues[j] << " is " << fResultLabel << ", DFvalue " << fResultValue << endl;
		}
	}

	// prediction sample
//	vector<float> descriptorsValues;
//	m_pPositiveValid->getDescriptorValuesFromImageAt(0, descriptorsValues);
//	Mat sampleMat = (Mat_<float>(1,1) << descriptorsValues[0]);
//	// single patch image:
//	//Mat sampleMat = (Mat_<float>(1,descriptorsValues.size()) << descriptorsValues);
//
//	float fResult = m_pSVM->predict(sampleMat, false);
//
//	cout << "SVM predict for " << descriptorsValues[0] << " is " << fResult << ", DFvalue " << m_pSVM->predict(sampleMat, true) << endl;

}

void mai::UDoMLDP::trainSVMOnDataSets(DataSet* positives, DataSet* negatives)
{
	Mat data(0, 0, CV_32FC1);;
	Mat labels(0, 0, CV_32SC1);

	//loading method:

	//setupTrainingData(positives, negatives, data, labels);

	setupTrainingDataForSinglePatchImage(positives, negatives, data, labels);

	std::string strDataname = "trainingdata";
	IOUtils::writeMatToCSV(data, strDataname);
	std::string strLabelname = "labeldata";
	IOUtils::writeMatToCSV(labels, strLabelname);

//	cout << data.size() << " - " << labels.size() << endl;
//	cout << data.at<float>(121103) << endl;
//	cout << (float)labels.at<uchar>(0) << endl;

	std::vector<std::vector<float> > vSupport;
	m_pSVM->trainSVM(data, labels, vSupport);

	std::string strSVMFilename = "trainedsvm.xml";
	m_pSVM->saveSVM(strSVMFilename);

//	cout << "Searching support vectors in positives .." << endl;
//
//	umSVM::searchSupportVector(positives, vSupport);
//
//	cout << "Searching support vectors in negatives .." << endl;
//
//	umSVM::searchSupportVector(negatives, vSupport);
//
//	cout << "Searching support vectors done." << endl;
}

void mai::UDoMLDP::setupTrainingData(DataSet* positives,
		DataSet* negatives,
		Mat &trainingData,
		Mat &labels)
{
	// Collect patches
	vector<float> vPositives;
	vector<int> vPositiveLabels;
	collectTrainingDataAndLabels(positives, vPositives, vPositiveLabels, 1.0);

	vector<float> vNegatives;
	vector<int> vNegativeLabels;
	collectTrainingDataAndLabels(negatives, vNegatives, vNegativeLabels, 0.0);

	vector<float> vData;
	vData.insert(std::end(vData), std::begin(vPositives), std::end(vPositives));
	vData.insert(std::end(vData), std::begin(vNegatives), std::end(vNegatives));

	vector<int> vLabels;
	vLabels.insert(std::end(vLabels), std::begin(vPositiveLabels), std::end(vPositiveLabels));
	vLabels.insert(std::end(vLabels), std::begin(vNegativeLabels), std::end(vNegativeLabels));

	// setup training matrices
	unsigned int iNumPatches = vPositives.size() + vNegatives.size();

	//	// rows, columns, type, datapointer
	//	Mat data( iNumPatches, 1, CV_32FC1, &vData[0]);
	//	Mat labels( iNumPatches, 1, CV_32SC1, &vLabels[0]);// CV_32FC1 not integral ??
	trainingData.create(iNumPatches, 1, CV_32FC1);
	labels.create(iNumPatches, 1, CV_32SC1);

	for(unsigned int i = 0; i < iNumPatches ; ++i)
	{
		trainingData.at<float>(i) = vData[i];
		labels.at<uchar>(i) = vLabels[i];
	}
}

void mai::UDoMLDP::collectTrainingDataAndLabels(DataSet* data,
		std::vector<float> &vTrainingData,
		std::vector<int> &vLabels,
		int iLabel)
{
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		vector<float> descriptorsValues;
		data->getDescriptorValuesFromImageAt(i, descriptorsValues);

		vTrainingData.insert(std::end(vTrainingData), std::begin(descriptorsValues), std::end(descriptorsValues));
		for(unsigned int j = 0; j < descriptorsValues.size(); ++j)
		{
			vLabels.push_back(iLabel);
		}
	}
}

void mai::UDoMLDP::setupTrainingDataForSinglePatchImage(DataSet* positives,
				DataSet* negatives,
				Mat &trainingData,
				Mat &labels)
{
	// Collect patches
	int iFeatureSizePos, iFeatureSizeNeg;
	vector<vector<float> > vPositives;
	vector<int> vPositiveLabels;
	iFeatureSizePos = collectTrainingDataAndLabelsForSingelPatchImage(positives, vPositives, vPositiveLabels, 1);

	vector<vector<float> > vNegatives;
	vector<int> vNegativeLabels;
	iFeatureSizeNeg = collectTrainingDataAndLabelsForSingelPatchImage(negatives, vNegatives, vNegativeLabels, 0);

	if(iFeatureSizeNeg != iFeatureSizePos)
	{
		cout << "[setupTrainingDataForSinglePatchImage] ERROR feature size for positive and negative do not match" << endl;
		return;
	}

	vector<vector<float> > vData;
	vData.insert(std::end(vData), std::begin(vPositives), std::end(vPositives));
	vData.insert(std::end(vData), std::begin(vNegatives), std::end(vNegatives));

	vector<int> vLabels;
	vLabels.insert(std::end(vLabels), std::begin(vPositiveLabels), std::end(vPositiveLabels));
	vLabels.insert(std::end(vLabels), std::begin(vNegativeLabels), std::end(vNegativeLabels));

	cout << "[setupTrainingDataForSinglePatchImage]  number of features: " << iFeatureSizePos << endl;

	// setup training matrices
	unsigned int iNumPatches = vPositives.size() + vNegatives.size();

	trainingData.create(iNumPatches, iFeatureSizePos, CV_32FC1);
	labels.create(iNumPatches, 1, CV_32SC1);
	cout << "[setupTrainingDataForSinglePatchImage]  Training matrix " << trainingData.rows << "x" << trainingData.cols << endl;
	cout << "[setupTrainingDataForSinglePatchImage]  label matrix " << labels.rows << "x" << labels.cols << endl;

	for(unsigned int i = 0; i < iNumPatches ; ++i)
	{
		vector<float> vCurrentData = vData[i];
		for(unsigned int j = 0; j < vCurrentData.size(); ++j)
		{
			trainingData.at<float>(i, j) = vCurrentData[j];
		}
		labels.at<int>(i) = vLabels[i];
	}

//	cout << trainingData.size() << " - " << labels.size() << endl;
//	cout << trainingData.at<float>(0, 121103) << " _ " << vData[0][121103] << endl;
//	cout << (float)labels.at<uchar>(0) << " l " << vLabels[0] << endl;

}

int mai::UDoMLDP::collectTrainingDataAndLabelsForSingelPatchImage(DataSet* data,
		std::vector<std::vector<float> > &vTrainingData,
		std::vector<int> &vLabels,
		int iLabel)
{
	unsigned int iDescriptorValueSize = 0;

	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		vector<float> descriptorsValues;
		data->getDescriptorValuesFromImageAt(i, descriptorsValues);

		if(iDescriptorValueSize == 0)
		{
			iDescriptorValueSize = descriptorsValues.size();
		}
		if (iDescriptorValueSize == descriptorsValues.size())
		{
			if(Constants::PCA_REDUCTION)
			{
				vector<float> reducedFeatures;

				umPCA::decreaseHOGDescriptorCellsByPCA(descriptorsValues,
						reducedFeatures,
						Constants::HOG_BINS);

				vTrainingData.push_back(reducedFeatures);
			}
			else
			{
				vTrainingData.push_back(descriptorsValues);
			}
			vLabels.push_back(iLabel);
		}
		else
		{
			cout << "[collectTrainingDataAndLabelsForSingelPatchImage] ERROR training data has to be uniform!" << endl;
		}
	}
	return iDescriptorValueSize;
}

bool isPositivePrediction(float svmPrediction) {
  // TODO Is this correct?
  return svmPrediction != 0;
}

int mai::UDoMLDP::predictDataSetbySVMForSinglePatchImage(DataSet* data)
{
  int numPositiveMatches = 0;
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		vector<float> descriptorsValues;
		data->getDescriptorValuesFromImageAt(i, descriptorsValues);

		if(Constants::PCA_REDUCTION)
		{
			vector<float> reducedFeatures;

			umPCA::decreaseHOGDescriptorCellsByPCA(descriptorsValues,
					reducedFeatures,
					Constants::HOG_BINS);

			descriptorsValues.clear();
			descriptorsValues = reducedFeatures;
		}

		// setup matrix
		Mat predictionData(1, descriptorsValues.size(), CV_32FC1, &descriptorsValues[0]);;

		if(Constants::DEBUG_MAIN_SVM)
		{
			cout << "[predictDataSetbySVMForSinglePatchImage]  prediction matrix " << predictionData.rows << "x" << predictionData.cols << endl;
		}

//		for(unsigned int j = 0; j < descriptorsValues.size(); ++j)
//		{
//			predictionData.at<float>(0, j) = descriptorsValues[j];
//		}

		float fResultLabel = m_pSVM->predict(predictionData, false);
		float fResultValue = m_pSVM->predict(predictionData, true);

		if(Constants::DEBUG_MAIN_SVM)
		{
		    cout << "[predictDataSetbySVMForSinglePatchImage] SVM predict for image " << i << " is " << fResultLabel << ", DFvalue " << fResultValue << endl;
		}

		if(isPositivePrediction(fResultLabel))
		  numPositiveMatches++;

		std::string winName;

		winName = isPositivePrediction(fResultLabel) ? "Pos" : "Neg";

		destroyWindow("Pos");
    destroyWindow("Neg");

    const Mat* image = data->getImageAt(i);
//    ImageDisplayUtils::displayImage(winName, *image, 50);

	}
  return numPositiveMatches;
}

void mai::UDoMLDP::predictWholeDataSetbySVMForSinglePatchImage(DataSet* data)
{
	// Collect patches
	int iFeatureSize;
	vector<vector<float> > vFeatures;
	vector<int> vDummy;
	iFeatureSize = collectTrainingDataAndLabelsForSingelPatchImage(data, vFeatures, vDummy, 1);

	// setup matrix
	unsigned int iNumPatches = vFeatures.size();

	Mat predictionData(iNumPatches, iFeatureSize, CV_32FC1);

	for(unsigned int i = 0; i < iNumPatches ; ++i)
	{
		vector<float> vCurrentData = vFeatures[i];
		for(unsigned int j = 0; j < vCurrentData.size(); ++j)
		{
			predictionData.at<float>(i, j) = vCurrentData[j];
		}
	}
	cout << "[predictWholeDataSetbySVMForSinglePatchImage]  prediction matrix " << predictionData.rows << "x" << predictionData.cols << endl;

	Mat results(iNumPatches, 1, CV_32SC1);

	m_pSVM->predict(predictionData, results);

	cout << "[predictWholeDataSetbySVMForSinglePatchImage]SVM predict result has  " << results.rows << " rows." << endl;

	for(int i = 0; i < results.rows; ++i)
	{
		cout << "[predictWholeDataSetbySVMForSinglePatchImage]SVM predict for image " << i << " is " << results.at<int>(i) << endl;
	}
}
