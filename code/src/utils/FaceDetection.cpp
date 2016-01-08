/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * faceDetection.cpp
 *
 *  Created on: Jan 7, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "FaceDetection.h"

#include "../data/DataSet.h"
#include "../Constants.h"
#include "../IO/IOUtils.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
using namespace mai;


mai::FaceDetection::FaceDetection()
{}

mai::FaceDetection::~FaceDetection()
{}

DataSet* mai::FaceDetection::detectFaces(DataSet* data,
		const std::string &strFilename,
		double dScale,
		int iMinNeighbors,
		Size minSize,
		Size maxSize)
{
	DataSet* faces = new DataSet();
	vector<Mat*> vImages;
	vector<string> vImageNames;
	CascadeClassifier cascade;

	if(!cascade.load(strFilename))
    {
        cerr << "[mai::FaceDetection::detectFaces] ERROR: Could not load classifier cascade from file " << strFilename << endl;
		return NULL;
	}

	// Extract faces from all images in dataset.
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		const Mat* image = data->getImageAt(i);
		Mat face;

		if(detectFace(*image,
				face,
				cascade,
				dScale,
				iMinNeighbors,
				minSize,
				maxSize))
		{
			Mat* pImage = new Mat(face);
			vImages.push_back(pImage);
			vImageNames.push_back(data->getImageNameAt(i));
		}
	}

	if(Constants::DEBUG_FACE_DETECTION)
	{
		string strPath = "outFD";

		IOUtils::writeImages(vImages,
				vImageNames,
				strPath);
	}

	faces->setImages(vImages, vImageNames);

	return faces;
}

bool mai::FaceDetection::detectFace(const Mat &image,
		Mat &face,
		CascadeClassifier cascade,
		double dScale,
		int iMinNeighbors,
		Size minSize,
		Size maxSize)
{
	vector<Rect> discoveredAreas;

	// TODO Why the scaling, does not make any difference ?
//	Mat smallImg( cvRound (image.rows/dScale), cvRound(image.cols/dScale), CV_8UC1 );//make the speed of detection fast
//	resize( image, smallImg, smallImg.size(),0,0,INTER_LINEAR );

//	cascade.detectMultiScale( smallImg, discoveredAreas, dScale, iMinNeighbors, 0 | CV_HAAR_SCALE_IMAGE, minSize, maxSize );
	cascade.detectMultiScale( image, discoveredAreas, dScale, iMinNeighbors, 0 | CV_HAAR_SCALE_IMAGE, minSize, maxSize );

	if(Constants::DEBUG_FACE_DETECTION)
	{
		cout << "[mai::FaceDetection::detectFace] faces found: " << discoveredAreas.size() << endl;
	}

	for( vector<Rect>::const_iterator it = discoveredAreas.begin(); it != discoveredAreas.end(); it++ )
	{
//		Rect rect(it->x*dScale, it->y*dScale, (it->x+it->width)*dScale-it->x*dScale,(it->y+it->height)*dScale-it->y);
		Rect rect(it->x, it->y, (it->x + it->width) - it->x, (it->y + it->height) - it->y);
		if(rect.height > 0 && rect.width > 0)
		{
			face = image(rect);
		}
	}

	if(!face.empty())
	{
		return true;
	}
	else
	{
		return false;
	}
}
