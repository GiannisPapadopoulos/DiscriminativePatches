/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * IOUtils.cpp
 *
 *  Created on: Nov 7, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "IOUtils.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

using namespace cv;

using namespace std;

mai::IOUtils::IOUtils()
{}
mai::IOUtils::~IOUtils()
{}

bool mai::IOUtils::loadImages( std::vector<Mat> &vImages, int iMode, const string &strDirectory )
{
	boost::filesystem::path directory( boost::filesystem::initial_path<boost::filesystem::path>() );
	directory = boost::filesystem::system_complete( boost::filesystem::path( strDirectory ) );

	if ( !exists( directory ) )
	{
		cout << "ERROR loading images. Path does not exist: " << directory << endl;
		return false;
	}
	cout << "Loading images from " << directory << endl;

	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	for ( boost::filesystem::directory_iterator itr( directory ); itr != end_itr; ++itr )
	{
		if ( !is_directory(itr->status()) )
		{
			//string strFilename = itr->path().leaf().string();
			Mat image = imread( itr->path().c_str(), iMode );

			if( !image.empty() )// Check for valid input
			{
				cout << "Image loaded successfully: " << itr->path() << endl;
				vImages.push_back ( image );
			}
			else
			{
				cout << "ERROR loading Image: " << itr->path() << endl;
			}
		}
	}
	return true;
}

void mai::IOUtils::convertImages( std::vector<Mat> &vImages, std::vector<Mat> &vConvertedImages, int iMode )
{
	for( Mat image : vImages)
	{
		Mat convertedImage;
		cvtColor(image, convertedImage, iMode);

		if( !convertedImage.empty() )// Check for valid input
		{
			vConvertedImages.push_back ( convertedImage );
		}
	}
}

void mai::IOUtils::getMaxImageDimensions( std::vector<Mat> &vImages, int &iMaxHeight, int &iMaxWidth )
{
	iMaxHeight = 0;
	iMaxWidth = 0;

	for( Mat image : vImages)
	{
		Size s = image.size();
		int iW = s.width;
		int iH = s.height;

		iMaxWidth = std::max(iMaxWidth, iW);
		iMaxHeight = std::max(iMaxHeight, iH);
	}
}

void mai::IOUtils::sampleImage( Mat &image, Mat &sampledImage, int iHeight, int iWidth )
{
	cv::Size s = image.size();
	int iW = s.width;
	int iH = s.height;

	if ( iW * iH == iHeight * iWidth )
		sampledImage = image;
	else
	{
		if ( iW * iH < iHeight * iWidth )
		{
			pyrUp( image, sampledImage, Size( iWidth, iHeight ) );
		}
		else
		{
			pyrDown( image, sampledImage, Size( iWidth, iHeight ) );
		}
	}

	assert( !sampledImage.empty() );
}

void mai::IOUtils::sampleImages( std::vector<Mat> &vImages, std::vector<Mat> &vSampledImages, int iHeight, int iWidth )
{
	for( Mat image : vImages)
	{
		Size s = image.size();
		int iW = s.width;
		int iH = s.height;

		Mat sampledImage;

		if ( iW * iH == iHeight * iWidth )
			sampledImage = image;
		else
		{
			if ( iW * iH < iHeight * iWidth )
			{
				pyrUp( image, sampledImage, Size( iWidth, iHeight ) );
			}
			else
			{
				pyrDown( image, sampledImage, Size( iWidth, iHeight ) );
			}
		}

		if( !sampledImage.empty() )// Check for valid input
		{
			vSampledImages.push_back ( sampledImage );
		}
	}
}

void mai::IOUtils::showImages( std::vector<Mat> &vImages )
{
	for( Mat image : vImages)
	{
		imshow("Image", image);
		waitKey(0);
	}
}



