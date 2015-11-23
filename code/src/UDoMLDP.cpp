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
	: m_pPositives(new DataSet())
	, m_pNegatives(new DataSet())

{}

mai::UDoMLDP::~UDoMLDP()
{
	delete m_pPositives;
	delete m_pNegatives;
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

	vector< float> descriptorsValues;
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
	cout << "Num positive images " << m_pPositives->setImages(images) << endl;

	int iMH, iMW;
	m_pPositives->getMaxDImensions(iMW, iMH);
	cout << "Max dimensions " << iMW << "x" << iMH << endl;

	images.clear();

	IOUtils::loadImages ( images, IMREAD_COLOR, strFilePathNegatives );
	cout << "Num negative images " << m_pNegatives->setImages(images) << endl;

	m_pNegatives->getMaxDImensions(iMW, iMH);
	cout << "Max dimensions " << iMW << "x" << iMH << endl;


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

	this->computeHOGForDataSet(m_pPositives,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			winStride,
			padding);
	this->computeHOGForDataSet(m_pNegatives,
			imageSize,
			blockSize,
			blockStride,
			cellSize,
			winStride,
			padding);


	trainSVM();

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

		vector< float> descriptorsValues;
		vector< Point> locations;

		cout << "[mai::UDoMLDP::computeHOGForDataSet] resizing image to " << imageSize << endl;
		Mat resizedImage;
		cv::resize(*image, resizedImage, imageSize);

		cvHOG::extractFeatures(descriptorsValues, resizedImage, blockSize, blockStride, cellSize, winStride, padding);

		data->addDescriptorValuesToImageAt(i, descriptorsValues);
	}
}

void mai::UDoMLDP::trainSVM()
{
	// Collect patches
	vector< float> vPositives;
	vector< float> vPositiveLabels;
	for(int i = 0; i < m_pPositives->getImageCount(); ++i)
	{
		vector< float> descriptorsValues;
		m_pPositives->getDescriptorValuesFromImageAt(i, descriptorsValues);

		vPositives.insert(std::end(vPositives), std::begin(descriptorsValues), std::end(descriptorsValues));
		for(unsigned int j = 0; j < descriptorsValues.size(); ++j)
		{
			vPositiveLabels.push_back(1.0);
		}
	}

	vector< float> vNegatives;
	vector< float> vNegativeLabels;
	for(int i = 0; i < m_pNegatives->getImageCount(); ++i)
	{
		vector< float> descriptorsValues;
		m_pNegatives->getDescriptorValuesFromImageAt(i, descriptorsValues);

		vNegatives.insert(std::end(vNegatives), std::begin(descriptorsValues), std::end(descriptorsValues));
		for(unsigned int j = 0; j < descriptorsValues.size(); ++j)
		{
			vNegativeLabels.push_back(0.0);
		}
	}

	vector< float> vData;
	vData.insert(std::end(vData), std::begin(vPositives), std::end(vPositives));
	vData.insert(std::end(vData), std::begin(vNegatives), std::end(vNegatives));
	vector< float> vLabels;
	vLabels.insert(std::end(vLabels), std::begin(vPositiveLabels), std::end(vPositiveLabels));
	vLabels.insert(std::end(vLabels), std::begin(vNegativeLabels), std::end(vNegativeLabels));

	// setup training matrices
	int iNumPatches = vPositives.size() + vNegatives.size();
	// rows, columns, type, datapointer
	Mat data( iNumPatches, 1, CV_32FC1, &vData[0]);
	Mat labels( iNumPatches, 1, CV_32SC1, &vLabels[0]);// CV_32FC1 not integral ??

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.gamma = 3;
	params.degree = 3;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM svm;
	svm.train(data, labels, Mat(), Mat(), params);

	// prediction sample
	vector< float> descriptorsValues;
	m_pPositives->getDescriptorValuesFromImageAt(0, descriptorsValues);
	Mat sampleMat = (Mat_<float>(1,1) << descriptorsValues[0]);

	cout << "SVM predict for " << descriptorsValues[0] << " is " << svm.predict(sampleMat) << ", DFvalue " << svm.predict(sampleMat, true) << endl;

	cout << "SVM support vector count: " << svm.get_support_vector_count() << ", vector0: " << svm.get_support_vector(0) << endl;
}



