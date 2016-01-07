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
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <iostream>

using namespace cv;
using namespace std;
using namespace mai;


mai::FaceDetection::FaceDetection(const string &strFilename)
{
	cout << "[mai::faceDetection::faceDetection] Loading cascade filter " << strFilename << endl;
//#ifdef linux
	m_Cascade = (CvHaarClassifierCascade*)cvLoad(strFilename.c_str(), 0, 0, 0);
//#else
//	m_Cascade = (CvHaarClassifierCascade*)cvLoad(strFilename.string(), 0, 0, 0);
//#endif

	cout << "[mai::faceDetection::faceDetection] Initialization done. " << endl;

}

mai::FaceDetection::~FaceDetection()
{
	delete m_Cascade;
}

DataSet* mai::FaceDetection::detectFaces(DataSet* data)
{
	DataSet* faces = new DataSet();
	vector<Mat*> vImages;
	vector<string> vImageNames;

	// Extract faces from all images in dataset.
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		const Mat* image = data->getImageAt(i);
		Mat face;

		detectFace(*image, face);

		Mat* pImage = new Mat(face);
		vImages.push_back(pImage);
		vImageNames.push_back(data->getImageNameAt(i));

	}

	faces->setImages(vImages, vImageNames);

	return faces;
}

void mai::FaceDetection::detectFace(const Mat &image,
		Mat &face)
{
	IplImage img = image;
	CvMemStorage* storage = cvCreateMemStorage(0);

	double scale=1.2;
	IplImage* small_img=cvCreateImage(cvSize(cvRound(image.cols/scale),cvRound(image.rows/scale)),8,1);
	IplImage* gray = cvCreateImage(cvSize(image.cols,image.rows),8,1);

	cvCvtColor(&img,gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR);
	cvEqualizeHist(small_img,small_img);
	cvClearMemStorage(storage);

	double t = (double)cvGetTickCount();
	CvSeq* objects = cvHaarDetectObjects(small_img,
			m_Cascade,
			storage,
			1.1,
			2,
			0,
			cvSize(20,20));

	t = (double)cvGetTickCount() - t;
	printf( "[mai::faceDetection::detectFace] Detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );

    //Loop through found objects and draw boxes around them
    for(int i=0;i<(objects? objects->total:0);++i)
    {
        CvRect* r=(CvRect*)cvGetSeqElem(objects,i);
		cvSetImageROI(&img,cvRect(r->x*1.15, r->y*1.15, (r->x+r->width*0.8), (r->y+r->height*0.8)));
    }

    face = &img;

    cvReleaseImage(&gray);
    cvReleaseImage(&small_img);

    imshow( "result", image );
	cvWaitKey(200);

}

