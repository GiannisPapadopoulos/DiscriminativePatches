#include "facedetection.h"
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

static CvMemStorage* storage = 0; 
static CvHaarClassifierCascade* cascade = 0;
const char* cascade_name = "C:/Users/apple/Desktop/Sophia/AI/Project/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt.xml"; 

using namespace cv;
using namespace std;
using namespace mai;

facedetection::facedetection(void)
{
}

facedetection::~facedetection(void)
{
}

void mai::facedetection::faceDetec(vector<Mat*> &vImages)
{
	std::vector<Mat*> images;
	for( Mat* image : images)//images)
	{

 cascade_name = "C:/Users/apple/Desktop/Sophia/AI/Project/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml"; 
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); 
 
    storage = cvCreateMemStorage(0); 
    cvNamedWindow( "result", 1 ); 

	double scale=1.2; 
    static CvScalar colors[] = { 
        {{0,0,255}},{{0,128,255}},{{0,255,255}},{{0,255,0}}, 
        {{255,128,0}},{{255,255,0}},{{255,0,0}},{{255,0,255}} 
    };//Just some pretty colors to draw with

    //Image Preparation 
    
	Mat image;
	IplImage* img=cvCloneImage(&(IplImage)image);

    IplImage* gray = cvCreateImage(cvSize(img->width,img->height),8,1); 
    IplImage* small_img=cvCreateImage(cvSize(cvRound(img->width/scale),cvRound(img->height/scale)),8,1); 
    cvCvtColor(img,gray, CV_BGR2GRAY); 
    cvResize(gray, small_img, CV_INTER_LINEAR);

    cvEqualizeHist(small_img,small_img); //Ö±·½Í¼¾ùºâ

    //Detect objects if any 
    // 
    cvClearMemStorage(storage); 
    double t = (double)cvGetTickCount(); 
    CvSeq* objects = cvHaarDetectObjects(small_img, 
                                                                        cascade, 
                                                                        storage, 
                                                                        1.1, 
                                                                        2, 
                                                                        0, 
                                                                        cvSize(20,20));

    t = (double)cvGetTickCount() - t; 
    printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );

    //Loop through found objects and draw boxes around them 
    for(int i=0;i<(objects? objects->total:0);++i) 
    { 
        CvRect* r=(CvRect*)cvGetSeqElem(objects,i); 
        cvRectangle(img, cvPoint(r->x*scale,r->y*scale), cvPoint((r->x+r->width)*scale,(r->y+r->height)*scale), colors[i%8]); 
		cvSetImageROI(img,cvRect(r->x*1.15, r->y*1.15, (r->x+r->width*0.8), (r->y+r->height*0.8)));
    } 

	
	cvWaitKey(200);
    cvShowImage( "result", img ); 
    cvReleaseImage(&gray); 
    cvReleaseImage(&small_img); 

    cvDestroyWindow("result"); 
	}
}