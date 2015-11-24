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

#include "svm/umSVM.h"
#include "data/DataSet.h"
#include "data/TrainingData.h"
#include "IO/IOUtils.h"
#include "featureExtraction/cvHOG.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>


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

void mai::UDoMLDP::unsupervisedDiscovery(std::string &strFilePathPositives, std::string &strFilePathNegatives)
{
	TrainingData* td = new TrainingData();

	//1. Load data

	std::vector<Mat*> images;

	IOUtils::loadImages ( images, IMREAD_COLOR, strFilePathPositives );

	td->setPositives(images);

	images.clear();

	IOUtils::loadImages ( images, IMREAD_COLOR, strFilePathNegatives );

	td->setNegatives(images);


	//	1.a discovery dataset D
	//	1.b natural world dataset N
	//2. Split datasets each in 2 equal sized disjoint parts {D1, D2}, {N1, N2}
	//3. Compute HOG descriptors for D1 at multiple resolutions ( at 7 different scales )

	Mat img = td->getPositives()[0];
	Size s = img.size();

	// image size has to be multiple of blocksize
	Mat resizedImage;
	cv::resize(img, resizedImage, Size(640,480));
	s = resizedImage.size();

	// OpenCV Documentation says that blocksize has to be 16x16 and cellsize 8x8. Other values are not supported.
	// Experiments say otherwise !?
	// blockssize and blockstride have to multiples of cellsize
	Size cellSize = Size(20,15);
	Size blockStride = Size(20,15);
	Size blockSize = Size(80,60);

	vector<float> descriptorsValues;
	cvHOG::extractFeatures(descriptorsValues, resizedImage, blockSize, blockStride, cellSize);

	Mat out;
	cvHOG::getHOGDescriptorVisualImage(out, resizedImage, descriptorsValues, s, cellSize, 2, 4.0);

	//show image
	cv::imshow("origin", out);
	cv::waitKey();

	//4. Take random sample patches S from D1 ( ~ 150 per image ) disallowing highly overlapping patches or patches without gradient energy (e.g. sky patches )
	//5. Compute clusters K from S using kmeans with high k = S/4

//	Mat data, labels;
//	td->getUniformTrainingData(data, labels);

	//6. Loop until convergence, i.e. top patches do not change ( 4 iterations ? ):
	//	6.1 Loop over all K(i) >= 3, skip small clusters
	//		6.1.a train svm on K(i) as positives and N1 as negatives -> classifiers Cnew(i)
	//		6.1.b hard mining ?
	//		6.1.c detect clusters Knew(i) from top m=5 firings for each detector by running Cnew(i) on validation dataset D2 for additional patches to prevent overfitting
	//
	//	6.2 Add Knew to clusters K
	//	6.3 Add Cnew to classifiers C
	//	6.4 Swap datasets D and N
	//7. Loop over all classifiers to compute ranking
	//	7.1 Compute purity: sum up svm detector scores of top r cluster members ( r > m )
	//	7.2 Compute discriminativness: ratio of firings for cluster on D and D u N ( rarely )
	//	7.3 sum up score
	//8. Select top n classifiers from above scores

	delete td;
}

void mai::UDoMLDP::basicDetecion(std::string &strFilePathPositives, std::string &strFilePathNegatives)
{
	std::vector<Mat*> images;

	IOUtils::loadImages ( images, IMREAD_COLOR, strFilePathPositives );
	cout << "Num positive images " << images.size() << endl;

	std::vector<Mat*> images2Half(std::make_move_iterator(images.begin() + images.size()/2), std::make_move_iterator(images.end()));
	images.erase(images.begin() + images.size()/2, images.end());

	m_pPositiveTrain->setImages(images);
	m_pPositiveValid->setImages(images2Half);

	int iMH, iMW;
	m_pPositiveTrain->getMaxDImensions(iMW, iMH);
	cout << "Num positive training images: " << m_pPositiveTrain->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;
	m_pPositiveValid->getMaxDImensions(iMW, iMH);
	cout << "Num positive validation images: " << m_pPositiveValid->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;

	images.clear();

	IOUtils::loadImages ( images, IMREAD_COLOR, strFilePathNegatives );
	cout << "Num negative images " << images.size() << endl;

	std::vector<Mat*> images2HalfNeg(std::make_move_iterator(images.begin() + images.size()/2), std::make_move_iterator(images.end()));
	images.erase(images.begin() + images.size()/2, images.end());

	m_pNegativeTrain->setImages(images);
	m_pNegativeValid->setImages(images2HalfNeg);

	m_pNegativeTrain->getMaxDImensions(iMW, iMH);
	cout << "Num negative training images: " << m_pNegativeTrain->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;
	m_pNegativeValid->getMaxDImensions(iMW, iMH);
	cout << "Num negative validation images: " << m_pNegativeValid->getImageCount() << ", Max dimensions: " << iMW << "x" << iMH << endl;


	// OpenCV Documentation says that blocksize has to be 16x16 and cellsize 8x8. Other values are not supported.
	// Experiments say otherwise !?
	// blockssize and blockstride have to multiples of cellsize
	// image size has to be multiple of blocksize
	Size cellSize = Size(20,15);
	Size blockStride = Size(20,15);
	Size blockSize = Size(80,60);
	Size imageSize = Size(640,480);

	// Not sure about these
	Size winStride = Size(0,0);
	Size padding = Size(0,0);

	this->computeHOGForDataSet(m_pPositiveTrain,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			winStride,
			padding);
	this->computeHOGForDataSet(m_pNegativeTrain,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			winStride,
			padding);
	this->computeHOGForDataSet(m_pPositiveValid,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			winStride,
			padding);
	this->computeHOGForDataSet(m_pNegativeValid,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			winStride,
			padding);

	trainSVMOnDataSets(m_pPositiveTrain, m_pNegativeTrain);

	// prediction sample
	// !! has to correspond with training data !!
	vector<float> descriptorsValues;
	m_pPositiveValid->getDescriptorValuesFromImageAt(0, descriptorsValues);
	Mat sampleMat = (Mat_<float>(1,1) << descriptorsValues[0]);
	// single patch image:
	//Mat sampleMat = (Mat_<float>(1,descriptorsValues.size()) << descriptorsValues);

	float fResult = m_pSVM->predict(sampleMat, false);

	cout << "SVM predict for " << descriptorsValues[0] << " is " << fResult << ", DFvalue " << m_pSVM->predict(sampleMat, true) << endl;

	//predictDataSetbySVM(m_pPositiveValid);

	//predictDataSetbySVM(m_pNegativeValid);

}

void mai::UDoMLDP::predictDataSetbySVM(DataSet* data)
{
	for(int i = 0; i < data->getImageCount(); ++i)
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
}

void mai::UDoMLDP::computeHOGForDataSet(DataSet* data,
		Size imageSize,
		Size blockSize,
		Size blockStride,
		Size cellSize,
		Size winStride,
		Size padding)
{
	for(int i = 0; i < data->getImageCount(); ++i)
	{
		const Mat* image = data->getImageAt(i);

		cout << "[mai::UDoMLDP::computeHOGForDataSet] resizing image to " << imageSize << endl;
		Mat resizedImage;
		cv::resize(*image, resizedImage, imageSize);

		vector< float> descriptorsValues;

		cvHOG::extractFeatures(descriptorsValues, resizedImage, blockSize, blockStride, cellSize, winStride, padding);

		data->addDescriptorValuesToImageAt(i, descriptorsValues);
	}
}

void mai::UDoMLDP::trainSVMOnDataSets(DataSet* positives, DataSet* negatives)
{
	Mat data(0, 0, CV_32FC1);;
	Mat labels(0, 0, CV_32SC1);

	//loading method:

	setupTrainingData(positives, negatives, data, labels);

	//setupTrainingDataForSinglePatchImage(positives, negatives, data, labels);

//	cout << data.size() << " - " << labels.size() << endl;
//	cout << data.at<float>(121103) << endl;
//	cout << (float)labels.at<uchar>(0) << endl;

	std::vector<float> vSupport;
	m_pSVM->trainSVM(data, labels, vSupport);

}

/**
 * Construct Mat for trainingdata.
 * Calls collectTrainingDataAndLabels for positives and negatives
 */
void mai::UDoMLDP::setupTrainingData(DataSet* positives,
		DataSet* negatives,
		Mat &trainingData,
		Mat &labels)
{
	// Collect patches
	vector<float> vPositives;
	vector<float> vPositiveLabels;
	collectTrainingDataAndLabels(positives, vPositives, vPositiveLabels, 1.0);

	vector<float> vNegatives;
	vector<float> vNegativeLabels;
	collectTrainingDataAndLabels(negatives, vNegatives, vNegativeLabels, 0.0);

	vector<float> vData;
	vData.insert(std::end(vData), std::begin(vPositives), std::end(vPositives));
	vData.insert(std::end(vData), std::begin(vNegatives), std::end(vNegatives));

	vector<float> vLabels;
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
		std::vector<float> &vLabels,
		float fLabel)
{
	for(int i = 0; i < data->getImageCount(); ++i)
	{
		vector<float> descriptorsValues;
		data->getDescriptorValuesFromImageAt(i, descriptorsValues);

		vTrainingData.insert(std::end(vTrainingData), std::begin(descriptorsValues), std::end(descriptorsValues));
		for(unsigned int j = 0; j < descriptorsValues.size(); ++j)
		{
			vLabels.push_back(fLabel);
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
	vector<float> vPositiveLabels;
	iFeatureSizePos = collectTrainingDataAndLabelsForSingelPatchImage(positives, vPositives, vPositiveLabels, 1.0);

	vector<vector<float> > vNegatives;
	vector<float> vNegativeLabels;
	iFeatureSizeNeg = collectTrainingDataAndLabelsForSingelPatchImage(negatives, vNegatives, vNegativeLabels, 0.0);

	if(iFeatureSizeNeg != iFeatureSizePos)
	{
		cout << "[setupTrainingDataForSinglePatchImage] ERROR feature size for positive and negative do not match" << endl;
		return;
	}

	vector<vector<float> > vData;
	vData.insert(std::end(vData), std::begin(vPositives), std::end(vPositives));
	vData.insert(std::end(vData), std::begin(vNegatives), std::end(vNegatives));

	vector<float> vLabels;
	vLabels.insert(std::end(vLabels), std::begin(vPositiveLabels), std::end(vPositiveLabels));
	vLabels.insert(std::end(vLabels), std::begin(vNegativeLabels), std::end(vNegativeLabels));

	// setup training matrices
	unsigned int iNumPatches = vPositives.size() + vNegatives.size();

	trainingData.create(iNumPatches, iFeatureSizePos, CV_32FC1);
	labels.create(iNumPatches, 1, CV_32SC1);

	for(unsigned int i = 0; i < iNumPatches ; ++i)
	{
		vector<float> vCurrentData = vData[i];
		for(unsigned int j = 0; j < vCurrentData.size(); ++j)
		{
			trainingData.at<float>(i, j) = vCurrentData[j];
		}
		labels.at<uchar>(i) = vLabels[i];
	}

//	cout << trainingData.size() << " - " << labels.size() << endl;
//	cout << trainingData.at<float>(0, 121103) << " _ " << vData[0][121103] << endl;
//	cout << (float)labels.at<uchar>(0) << " l " << vLabels[0] << endl;

}

int mai::UDoMLDP::collectTrainingDataAndLabelsForSingelPatchImage(DataSet* data,
		std::vector<std::vector<float> > &vTrainingData,
		std::vector<float> &vLabels,
		float fLabel)
{
	unsigned int iDescriptorValueSize = 0;

	for(int i = 0; i < data->getImageCount(); ++i)
	{
		vector<float> descriptorsValues;
		data->getDescriptorValuesFromImageAt(i, descriptorsValues);

		if(iDescriptorValueSize == 0)
		{
			iDescriptorValueSize = descriptorsValues.size();
		}
		if (iDescriptorValueSize == descriptorsValues.size()) {
			vTrainingData.push_back(descriptorsValues);
			vLabels.push_back(fLabel);
		}
		else
		{
			cout << "[collectTrainingDataAndLabelsForSingelPatchImage] ERROR training data has to be uniform!" << endl;
		}
	}
	return iDescriptorValueSize;
}
