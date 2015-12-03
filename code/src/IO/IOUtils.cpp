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
#include "../Constants.h"
#include "../data/DataSet.h"
#include "../featureExtraction/cvHOG.h"

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

bool mai::IOUtils::loadImages( std::vector<Mat*> &vImages, int iMode, const string &strDirectory, bool bEqualize )
{
	boost::filesystem::path directory( boost::filesystem::initial_path<boost::filesystem::path>() );
	directory = boost::filesystem::system_complete( boost::filesystem::path( strDirectory ) );

	if ( !exists( directory ) )
	{
		cout << "ERROR loading images. Path does not exist: " << directory << endl;
		return false;
	}
	if(Constants::DEBUG_IMAGE_LOADING) {
	  cout << "Loading images from " << directory << endl;
	}

	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	for ( boost::filesystem::directory_iterator itr( directory ); itr != end_itr; ++itr )
	{
		if ( !is_directory(itr->status()) )
		{
			//string strFilename = itr->path().leaf().string();
#ifdef linux
			Mat image = imread( itr->path().c_str(), iMode );
#else
			Mat image = imread( itr->path().string(), iMode );
#endif
			if( !image.empty() )// Check for valid input
			{
				if(Constants::DEBUG_IMAGE_LOADING) {
					cout << "Image loaded successfully: " << itr->path() << endl;
				}

				Mat* pImage;
				if (bEqualize) {
					cvtColor(image, image, CV_BGR2GRAY);
					equalizeHist(image, image);
				}
				pImage = new Mat(image);
				vImages.push_back(pImage);
			}
			else
			{
				cout << "ERROR loading Image: " << itr->path() << endl;
			}
		}
	}
	return true;
}

void mai::IOUtils::addFlippedImages( std::vector<Mat*> &vImages, int iFlipMode )
{
	std::vector<Mat*> vDoubledImages;

	for( Mat* image : vImages)
	{
		Mat flippedImage;
		flip(*image, flippedImage, iFlipMode);

		if( !flippedImage.empty() )// Check for valid input
		{
			Mat* pImage = new Mat(flippedImage);
			vDoubledImages.push_back(pImage);
		}
	}

	vImages.insert(std::end(vImages), std::begin(vDoubledImages), std::end(vDoubledImages));
}

void mai::IOUtils::addDistortedImages( std::vector<Mat*> &vImages, int variationsOfEachImage )
{
  std::vector<Mat*> vDistortedImages;

  for( Mat* image : vImages)
  {
    for(int i = 0; i < variationsOfEachImage; i++) {
//      Mat distortedImage(image->rows, image->cols, CV_8U);
      Mat distortedImage = image->clone();

      cout << i << endl;

     for(int row = 0; row < image->rows; row++) {
       for(int col = 0; col < image->cols; col++) {
         double pixelValue = image->at<double>(row, col);
         distortedImage.at<double>(row, col) = pixelValue;
       }
     }

     // Also crashes
     // showImage(distortedImage);

      if( !distortedImage.empty() )// Check for valid input
      {
        Mat* pImage = new Mat(distortedImage);
        // Crashes with segmentation fault
        vDistortedImages.push_back(pImage);
      } else {
        cout << "empty " << endl;
      }
    }
  }
  cout << "original " << vImages.size() << " dist " << vDistortedImages.size() << endl;
  vImages.insert(std::end(vImages), std::begin(vDistortedImages), std::end(vDistortedImages));

}

void mai::IOUtils::convertImages( std::vector<Mat*> &vImages, std::vector<Mat*> &vConvertedImages, int iMode )
{
	for( Mat* image : vImages)
	{
		Mat convertedImage;
		cvtColor(*image, convertedImage, iMode);

		if( !convertedImage.empty() )// Check for valid input
		{
			Mat* pImage = new Mat(convertedImage);
			vConvertedImages.push_back(pImage);
		}
	}
}

void mai::IOUtils::equalizeImages( std::vector<Mat*> &vImages, std::vector<Mat*> &vConvertedImages )
{
	for( Mat* image : vImages)
	{
		Mat convertedImage;
		equalizeHist(*image, convertedImage);

		if( !convertedImage.empty() )// Check for valid input
		{
			Mat* pImage = new Mat(convertedImage);
			vConvertedImages.push_back ( pImage );
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
		showImage( image );
	}
}

void mai::IOUtils::showImage( Mat &image )
{
	imshow("Image", image);
	waitKey(0);
}

void mai::IOUtils::writeImages( std::vector<cv::Mat*> &vImages, std::string &strPath, std::string &strFileNameBase )
{
	for( unsigned int i = 0; i < vImages.size(); ++i)
	{
		Mat image = *vImages[i];

		std::stringstream sstm;
		sstm << strPath << "/" << strFileNameBase << "_" << i << ".jpg";
		std::string strFileName = sstm.str();

		imwrite(strFileName, image);
	}
}

void mai::IOUtils::writeHOGImages( mai::DataSet* data,
			std::string &strPath,
			std::string &strFileNameBase,
			cv::Size imageSize,
			cv::Size cellSize,
			int scaleFactor,
			double vizFactor)
{
	for ( unsigned int i = 0; i < data->getImageCount(); ++i )
	{
		const Mat* image = data->getImageAt(i);
		std::vector<float> vDescriptorValues;
		data->getDescriptorValuesFromImageAt(i, vDescriptorValues);

		if (vDescriptorValues.size() <= 0 || image == NULL) {
			continue;
		}

		Mat resizedImage, outImage;
		resize(*image, resizedImage, imageSize);
		cvtColor(resizedImage, resizedImage, CV_GRAY2BGR);

		cvHOG::getHOGDescriptorVisualImage(outImage,
				resizedImage,
				vDescriptorValues,
				imageSize,
				cellSize,
				scaleFactor,
				vizFactor);

		std::stringstream sstm;
		sstm << strPath << "/" << strFileNameBase << "_" << i << ".jpg";
		std::string strFileName = sstm.str();

		imwrite(strFileName, outImage);
	}

}

