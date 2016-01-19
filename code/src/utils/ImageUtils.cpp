/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * ImageUtils.cpp
 *
 *  Created on: Jan 19, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "ImageUtils.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <boost/algorithm/string.hpp>

using namespace cv;

using namespace std;

mai::ImageUtils::ImageUtils()
{}

mai::ImageUtils::~ImageUtils()
{}

void mai::ImageUtils::addFlippedImages(vector<Mat*> &vImages,
		std::vector<std::string> &vImageNames,
		const int iFlipMode)
{
	vector<Mat*> vDoubledImages;

	for(int i = 0; i < vImages.size(); ++i)
	{
		Mat* image = vImages.at(i);
		Mat flippedImage;
		flip(*image, flippedImage, iFlipMode);

		// Check for valid input
		if( !flippedImage.empty() )
		{
			Mat* pImage = new Mat(flippedImage);
			vDoubledImages.push_back(pImage);

			// Add new name for new image
			std::vector<std::string> strs;
			boost::split(strs, vImageNames.at(i), boost::is_any_of("."));

			std::stringstream ss;
			for(int j = 0; j < strs.size(); ++j)
			{
				if(j == strs.size() - 1)
				{
					ss << "_flipped." << strs.at(j);
				}
				else
				{
					ss << strs.at(j);
				}
			}
			vImageNames.push_back(ss.str());
		}
	}

	vImages.insert(end(vImages), begin(vDoubledImages), end(vDoubledImages));
}

void mai::ImageUtils::convertImages(const vector<Mat*> &vImages,
		vector<Mat*> &vConvertedImages,
		const int iMode )
{
	for( Mat* image : vImages)
	{
		Mat convertedImage;
		cvtColor(*image, convertedImage, iMode);

		// Check for valid input
		if( !convertedImage.empty() )
		{
			Mat* pImage = new Mat(convertedImage);
			vConvertedImages.push_back(pImage);
		}
	}
}

void mai::ImageUtils::equalizeImages(const vector<Mat*> &vImages,
		vector<Mat*> &vConvertedImages )
{
	for( Mat* image : vImages)
	{
		Mat convertedImage;
		equalizeHist(*image, convertedImage);

		// Check for valid input
		if( !convertedImage.empty() )
		{
			Mat* pImage = new Mat(convertedImage);
			vConvertedImages.push_back ( pImage );
		}
	}
}

void mai::ImageUtils::getMaxImageDimensions(const vector<Mat> &vImages,
		int &iMaxHeight,
		int &iMaxWidth)
{
	iMaxHeight = 0;
	iMaxWidth = 0;

	for( Mat image : vImages)
	{
		Size s = image.size();
		int iW = s.width;
		int iH = s.height;

		iMaxWidth = max(iMaxWidth, iW);
		iMaxHeight = max(iMaxHeight, iH);
	}
}

void mai::ImageUtils::sampleImage(const Mat &image,
		Mat &sampledImage,
		const int iHeight,
		const int iWidth )
{
	Size s = image.size();
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

void mai::ImageUtils::sampleImages(const vector<Mat> &vImages,
		vector<Mat> &vSampledImages,
		const int iHeight,
		const int iWidth )
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
