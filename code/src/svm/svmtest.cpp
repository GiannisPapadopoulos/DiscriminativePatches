/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * svmtest.cpp
 *
 *  Created on: Nov 24, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "svmtest.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "../data/GenerateTestData.h"
#include "umSVM.h"


using namespace cv;

using namespace std;

using namespace mai;

SVMTest::SVMTest()
{}

SVMTest::~SVMTest()
{}

void SVMTest::testGeneratedTestData()
{
	int iImgSize = 400;
	int iNumImages = 2;
	Mat data( iNumImages, iImgSize*iImgSize, CV_32FC1 );
	Mat labels( iNumImages, 1, CV_32SC1 );// CV_32FC1 not integral ??

	GenerateTestData* td = new GenerateTestData();
	td->GenTestMatrix(data, labels, iImgSize, iNumImages);

	Mat sample( iNumImages * 2, iImgSize*iImgSize, CV_32FC1 );
	Mat sample_labels( iNumImages * 2, 1, CV_32FC1 );
	td->GenTestMatrix(sample, sample_labels, iImgSize, iNumImages * 2);

	Mat pos = Mat::ones( iImgSize, iImgSize, CV_32FC1 );
	td->DrawFilledCircle(pos);

	Mat neg = Mat::zeros( iImgSize, iImgSize, CV_32FC1 );
	td->DrawFilledRectangle(neg);

	delete td;

	Mat greyImage;
	cvtColor(data, greyImage, COLOR_GRAY2RGB);
	imwrite("trainingmatrix.png", greyImage);
	imshow("Example", data);
	waitKey(0);

	imwrite("pos.png", pos);
	imshow("Example", pos);
	waitKey(0);

	imwrite("neg.png", neg);
	imshow("Example", neg);
	waitKey(0);

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.gamma = 3;
	params.degree = 3;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(data, labels, Mat(), Mat(), params);

	//mai::SVM* mysvm = new mai::SVM();
	//mysvm->train(&data, &labels, &params);

	//float fRes = SVM.predict(pos.reshape(0, 1), true);

	cout << SVM.predict(pos.reshape(0, 1)) << endl;
	cout << SVM.predict(pos.reshape(0, 1), true) << endl;

	cout << SVM.get_support_vector_count() << endl;
	cout << SVM.get_support_vector(0) << endl;

	const float* fSV =  SVM.get_support_vector(0);

	Mat sv = Mat::ones( iImgSize, iImgSize, CV_32FC1 );
	for (int i = 0; i < sv.rows; ++i)
		{
			for (int j = 0; j < sv.cols; ++j)
			{
				sv.at<float>(i, j) = fSV[i * j + j];
			}
		}

	imshow("spv", sv);
	waitKey(0);
}

void mai::SVMTest::test1()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[4] = {1.0, -1.0, -1.0, -1.0};
	Mat labelsMat(4, 1, CV_32FC1, labels);

	float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	std::vector<std::vector<float> > supportVectors;

	//params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	umSVM* svm = new umSVM();
	int supportVectorCount = svm->trainSVM(trainingDataMat, labelsMat, supportVectors);

	cout << supportVectorCount << endl;
	cout << supportVectors[0][0] << endl;
	cout << supportVectors[0][1] << endl;

	Vec3b green(0,255,0), blue (255,0,0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1,2) << j,i);
			float response = svm->predict(sampleMat);//SVM.predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(i,j)  = green;
			else if (response == -1)
				image.at<Vec3b>(i,j)  = blue;

//			Mat res;
//			this->m_pSVM->predict(sampleMat, res);
//			if (res.at<float>(0) == 1)
//			    image.at<Vec3b>(i, j) = green;
//			else if (res.at<float>(0) == -1)
//			    image.at<Vec3b>(i, j) = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
	circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType  = 8;

	for (int i = 0; i < supportVectorCount; ++i)
	{
		std::vector<float> v = supportVectors[i];
		//const float* v = supportVectors[i];
		circle( image,  Point( (int) v[0], (int) v[1] ),   6,  Scalar(128, 128, 128), thickness, lineType);
	}

	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
}

#define NTRAINING_SAMPLES   100         // Number of training samples per class
#define FRAC_LINEAR_SEP     0.9f        // Fraction of samples which compose the linear separable part

void mai::SVMTest::test2()
{
	// Data for visual representation
	const int WIDTH = 512, HEIGHT = 512;
	Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	//--------------------- 1. Set up training data randomly ---------------------------------------
	Mat trainData(2*NTRAINING_SAMPLES, 2, CV_32FC1);
	Mat labels   (2*NTRAINING_SAMPLES, 1, CV_32FC1);

	RNG rng(100); // Random value generation class

	// Set up the linearly separable part of the training data
	int nLinearSamples = (int) (FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

	// Generate random points for the class 1
	Mat trainClass = trainData.rowRange(0, nLinearSamples);
	// The x coordinate of the points is in [0, 0.4)
	Mat c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1,2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

	// Generate random points for the class 2
	trainClass = trainData.rowRange(2*NTRAINING_SAMPLES-nLinearSamples, 2*NTRAINING_SAMPLES);
	// The x coordinate of the points is in [0.6, 1]
	c = trainClass.colRange(0 , 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1,2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

	//------------------ Set up the non-linearly separable part of the training data ---------------

	// Generate random points for the classes 1 and 2
	trainClass = trainData.rowRange(  nLinearSamples, 2*NTRAINING_SAMPLES-nLinearSamples);
	// The x coordinate of the points is in [0.4, 0.6)
	c = trainClass.colRange(0,1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1,2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

	//------------------------- Set up the labels for the classes ---------------------------------
	labels.rowRange(                0,   NTRAINING_SAMPLES).setTo(1);  // Class 1
	labels.rowRange(NTRAINING_SAMPLES, 2*NTRAINING_SAMPLES).setTo(2);  // Class 2

	//------------------------ 2. Set up the support vector machines parameters --------------------
//	CvSVMParams params;
//	params.svm_type    = SVM::C_SVC;
//	params.C           = 0.1;
//	params.kernel_type = SVM::LINEAR;
//	params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

	//------------------------ 3. Train the svm ----------------------------------------------------
	cout << "Starting training process" << endl;
//	CvSVM svm;
//	svm.train(trainData, labels, Mat(), Mat(), params);

	umSVM* svm = new umSVM();

	std::vector<std::vector<float> > supportVectors;
	int supportVectorCount = svm->trainSVM(trainData, labels, supportVectors);
	cout << supportVectorCount << endl;
	cout << supportVectors[0][0] << endl;
	cout << supportVectors[0][1] << endl;

	cout << "Finished training process" << endl;

	//------------------------ 4. Show the decision regions ----------------------------------------
	Vec3b green(0,100,0), blue (100,0,0);
	for (int i = 0; i < I.rows; ++i)
		for (int j = 0; j < I.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1,2) << i, j);
			float response = svm->predict(sampleMat);

			if      (response == 1)    I.at<Vec3b>(j, i)  = green;
			else if (response == 2)    I.at<Vec3b>(j, i)  = blue;
		}

	//----------------------- 5. Show the training data --------------------------------------------
	int thick = -1;
	int lineType = 8;
	float px, py;
	// Class 1
	for (int i = 0; i < NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i,0);
		py = trainData.at<float>(i,1);
		circle(I, Point( (int) px,  (int) py ), 3, Scalar(0, 255, 0), thick, lineType);
	}
	// Class 2
	for (int i = NTRAINING_SAMPLES; i <2*NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i,0);
		py = trainData.at<float>(i,1);
		circle(I, Point( (int) px, (int) py ), 3, Scalar(255, 0, 0), thick, lineType);
	}

	//------------------------- 6. Show support vectors --------------------------------------------
	thick = 2;
	lineType  = 8;

	for (int i = 0; i < supportVectorCount; ++i)
	{
		std::vector<float> v = supportVectors[i];
		///const float* v = supportVectors[i];
		circle( I,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thick, lineType);
	}

	imwrite("result.png", I);                      // save the Image
	imshow("SVM for Non-Linear Training Data", I); // show it to the user
	waitKey(0);
}
