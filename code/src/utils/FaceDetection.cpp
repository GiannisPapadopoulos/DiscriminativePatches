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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
using namespace mai;


mai::FaceDetection::FaceDetection(const string &strFilename)
{
	cout << "[mai::faceDetection::faceDetection] Loading cascade filter " << strFilename << endl;
//#ifdef linux
	//m_Cascade = (CascadeClassifier)load(strFilename.c_str(), 0, 0, 0);
//#else
//	m_Cascade = (CvHaarClassifierCascade*)cvLoad(strFilename.string(), 0, 0, 0);
//#endif

	cout << "[mai::faceDetection::faceDetection] Initialization done. " << endl;

}

mai::FaceDetection::~FaceDetection()
{
	//delete m_Cascade;
}

DataSet* mai::FaceDetection::detectFaces(DataSet* data)
{
	DataSet* faces = new DataSet();
	vector<Mat*> vImages;
	vector<string> vImageNames;
	CascadeClassifier cascade;
	string cascadeName = "C:/Users/apple/Desktop/Sophia/AI/Project/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml" ;

	if( !cascade.load(cascadeName))//从指定的文件目录中加载级联分类器
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
		//return 0;
	}

	// Extract faces from all images in dataset.
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		const Mat* image = data->getImageAt(i);
		Mat face;

		detectFace(*image, face,cascade);

		Mat* pImage = new Mat(face);
		vImages.push_back(pImage);
		vImageNames.push_back(data->getImageNameAt(i));

		waitKey(50);
		imshow( "result", face);

	}

	faces->setImages(vImages, vImageNames);

	return faces;
}

void mai::FaceDetection::detectFace(const Mat &image,
		Mat &face,CascadeClassifier cascade)
{
	
		int i = 0;
		double scale=1.2;
		vector<Rect> facess;
		double t = (double)cvGetTickCount();
		
		Mat gray,smallImg( cvRound (image.rows/scale), cvRound(image.cols/scale), CV_8UC1 );//make the speed of detection fast
		//cvtColor( face, face, CV_BGR2GRAY );
		resize( image, smallImg, smallImg.size(),0,0,INTER_LINEAR );
		equalizeHist( smallImg, smallImg );

		cascade.detectMultiScale( smallImg, facess,1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(150, 150) );

		t = (double)cvGetTickCount() - t;
		printf( "[mai::faceDetection::detectFace] Detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );

		//Loop through found objects and draw boxes around them
		for( vector<Rect>::const_iterator r = facess.begin(); r != facess.end(); r++, i++ )
		{
			vector<Rect> nestedObjects;
			Rect rect(r->x*scale, r->y*scale, (r->x+r->width)*scale-r->x*scale,(r->y+r->height)*scale-r->y);
			face = image(rect);
		}
		
		
}

