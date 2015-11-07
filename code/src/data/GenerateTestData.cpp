/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * GenerateTestData.cpp
 *
 *  Created on: Oct 11, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "GenerateTestData.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace cv;

namespace mai {

GenerateTestData::GenerateTestData()
{
	/* initialize random seed: */
	srand (time(NULL));
}

GenerateTestData::~GenerateTestData()
{}

void GenerateTestData::ShowST()
{
	int iImgSize = 400;
	int iNumImages = 40;
	Mat data( iNumImages, iImgSize*iImgSize, CV_32FC1 );
	Mat labels( iNumImages, 1, CV_8UC3 );

	GenTestMatrix(data, labels, iImgSize, iNumImages);

	imshow("Example", data);
	waitKey(0);

/*
	int iW = 400;
	Mat img = Mat::zeros( iW, iW, CV_8UC3 );

	DrawFilledCircle(img);

	imshow("Example", img);
	waitKey(0);

	img = Mat::zeros( iW, iW, CV_8UC3 );

	DrawFilledRectangle(img);

	imshow("Example", img);
	waitKey(0);
*/
}

void GenerateTestData::GenTestMatrix(Mat data, Mat labels, int iImgSize, int iNumImages)
{
	for ( int i = 0; i < iNumImages / 2; ++i)
	{
		Mat image = Mat::ones( iImgSize, iImgSize, CV_8UC1 );//CV_8UC3
		DrawFilledCircle(image);

		//imshow("Example", image);
		//waitKey(0);

		AddImageToData(data, image, i);
		labels.at<uchar>(i) = 1;
	}

	for ( int i = iNumImages / 2; i < iNumImages ; ++i)
	{
		Mat image = Mat::zeros( iImgSize, iImgSize, CV_8UC1 );
		DrawFilledRectangle(image);

		//imshow("Example", image);
		//waitKey(0);

		AddImageToData(data, image, i);
		labels.at<uchar>(i) = -1;
	}
}

void GenerateTestData::AddImageToData(Mat data, Mat image, int iPos)
{
	int iDataCol = 0;
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			data.at<float>(iPos, iDataCol++) = image.at<uchar>(i,j);
		}
	}
}

void GenerateTestData::DrawFilledCircle(Mat img)
{
	int iThickness = -1;//filled
	int iLineType = 8;

	Size s = img.size();
	int iW = s.width;

	rectangle( img, Point(0, 0), Point(iW, iW), Scalar( 255, 255, 255 ), iThickness, iLineType );

	int iRadius = 0;

	while (iRadius < (iW/6))
		iRadius = rand() % (iW/3);

	int iX = rand() % (iW/3) + iW/3;
	int iY = rand() % (iW/3) + iW/3;
//	int v3 = rand() % 30 + 1985;   // v3 in the range 1985-2014

	circle( img, Point(iX, iY), iRadius, Scalar( 0,0,0 ), iThickness, iLineType );
}

void GenerateTestData::DrawFilledRectangle(Mat img)
{
	int iThickness = -1;//filled
	int iLineType = 8;

	Size s = img.size();
	int iW = s.width;

	rectangle( img, Point(0, 0), Point(iW, iW), Scalar( 255, 255, 255 ), iThickness, iLineType );

	int iHeigth = 0;
	int iWidth = 0;

	while (iHeigth < (iW/6))
		iHeigth = rand() % (iW/3);
	while (iWidth < (iW/6))
		iWidth = rand() % (iW/3);

	int iX = rand() % (iW/3);
	int iY = rand() % (iW/3);

	rectangle( img, Point(iX, iY), Point(iX + iWidth, iY + iHeigth), Scalar( 0,0,0 ), iThickness, iLineType );
}

}  // namespace mai
