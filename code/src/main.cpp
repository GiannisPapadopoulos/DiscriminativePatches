/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * cvHOG.h
 *
 *  Created on: Nov 18, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include <iostream>
#include <string>

#include "configuration/Configuration.h"
#include "imageCatalog/CatalogClassificationSVM.h"
#include "imageCatalog/CatalogTraining.h"
#include "configuration/Constants.h"

using namespace std;
using namespace mai;

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>

using namespace cv;

typedef std::chrono::high_resolution_clock Clock;


void livePrediction(CatalogClassificationSVM* classifier)
{
	VideoCapture cap(0); // open the video camera no. 0

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	double dRectSize = 96.0;
	Size patchSize = classifier->getConfig()->getImageSize();

	cout << "[livePrediction] Frame size : " << dWidth << " x " << dHeight  << ", patchsize " << patchSize << endl;

	namedWindow("LivePrediciton", CV_WINDOW_AUTOSIZE);
	Rect rect = Rect(dWidth/2.0 - dRectSize/2.0, 2 * dHeight/3.0, dRectSize, dRectSize);

	string strClassifiedAs = "undefined";
	int fontFace = FONT_HERSHEY_TRIPLEX;
	double fontScale = 1.5;
	int thickness = 2;
	int baseline=0;

	map<string, float> mResults;

    auto timer = Clock::now();

	while(true)
	{
		Mat frame;

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "[livePrediction] ERROR! Cannot read a frame from video stream" << endl;
			break;
		}

		rectangle(frame, rect, Scalar( 0, 0, 255 ), 2, 8);

		// Check emotion every LIVE_TICK
	    auto action = Clock::now();
	    if((action - timer).count() % Constants::LIVE_TICK == 0)
		{
			Mat cut = frame(rect);
			Mat resized;

			resize(cut, resized, patchSize);
			strClassifiedAs = classifier->predict(resized, mResults);

			cout << "[livePrediction] Classified as " << strClassifiedAs << endl;
//			imshow("cut", resized);

			mResults.clear();
		}

		stringstream sstm;
		sstm << "You feel " << strClassifiedAs;
		string strText = sstm.str();

		Size textSize = getTextSize(strText, fontFace,
				fontScale, thickness, &baseline);
		// then put the text itself
		putText(frame, strText, Point(1* dWidth/20.0, 2* dHeight/10.0), fontFace, fontScale,
				Scalar( 0, 0, 255 ), thickness, 8);

		imshow("LivePrediciton", frame);

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "[livePrediction] esc key is pressed by user" << endl;
			break;
		}
	}
}


/**
 * This is just the beginning ..
 */
int main(int argc, char** argv )
{
	if(argc < 2)
	{
		cerr << "The application has 3 operating modes defined in the configuration file:" << endl
			<< "\t1. Train SVMs on a categorized image cataloge." << endl
			<< "\t2. Retrain SVMs on a categorized image cataloge." << endl
			<< "\t3. Classify images using trained SVMs." << endl
			<< "\t4. Live classification using local webcam." << endl
			<< "Usage:" << endl
			<< argv[0] << " [options]" << endl
			<< "options:" << endl
			<< "\t-config <config file>\tname and path to a configuration file defining application settings." << endl;
		return -1;
	}

	string strConfigFile = "";

	for(int i = 0; i < argc; ++i)
	{
		if(string(argv[i]) == "-config")
		{
			if(i+1 < argc)
			{
				strConfigFile = string(argv[i+1]);
			}
			else
			{
				cerr << "ERROR: Option -config given without filepath." << endl;
				return -1;
			}
		}
	}

	if(strConfigFile.empty())
	{
		cerr << "ERROR: No configuration file given." << endl;
		return -1;
	}

	Configuration* config = new Configuration(strConfigFile);

	if(config->getApplicationMode() == Configuration::appMode::Undef)
	{
		cout << "ERROR: Application mode undefined. What should I do ?" << endl;
		return -1;
	}

	if(config->getApplicationMode() == Configuration::appMode::Train
			|| config->getApplicationMode() == Configuration::appMode::Retrain)
	{
		cout << "[Main] Training classifiers according to configuration given in " << strConfigFile << endl;
		CatalogTraining* trainer = new CatalogTraining(config);
		trainer->processPipeline();
		delete trainer;
	}

	if(config->getApplicationMode() == Configuration::appMode::Predict)
	{
		cout << "[Main] Predicting image according to configuration given in " << strConfigFile << endl;
		CatalogClassificationSVM* classifier = new CatalogClassificationSVM(config);
		classifier->loadAndPredict();
		delete classifier;
	}

	if(config->getApplicationMode() == Configuration::appMode::Live)
	{
		cout << "[Main] Predicting image according to configuration given in " << strConfigFile << endl;
		CatalogClassificationSVM* classifier = new CatalogClassificationSVM(config);

		string strSVMPath = config->getSVMInputPath();

		cout << "[Main] Loading classifiers from " << strSVMPath << endl;

		if(!classifier->loadSVMs(strSVMPath))
		{
			cout << "[Main] ERROR! Loading svms " << strSVMPath << endl;
			return -1;
		}

		livePrediction(classifier);

		delete classifier;
	}

	delete config;

	return 0;
}

