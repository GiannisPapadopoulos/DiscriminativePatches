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
#include "../featureExtraction/umHOG.h"
#include "../svm/umSVM.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>

using namespace cv;

using namespace std;

mai::IOUtils::IOUtils()
{}
mai::IOUtils::~IOUtils()
{}

bool mai::IOUtils::loadCatalogue(map<string, DataSet* > &mCatalogue,
				int iCVLoadMode,
				const string &strDirectory,
				bool bAddFlipped,
				bool bEqualize )
{
	boost::filesystem::path directory( boost::filesystem::initial_path<boost::filesystem::path>() );
	directory = boost::filesystem::system_complete( boost::filesystem::path( strDirectory ) );

	if ( !exists( directory ) )
	{
		cout << "[mai::IOUtils::loadCatalogue] ERROR loading images. Path does not exist: " << directory << endl;
		return false;
	}

	// default construction yields past-the-end
	boost::filesystem::directory_iterator end_itr;

	for ( boost::filesystem::directory_iterator itr( directory ); itr != end_itr; ++itr )
	{
		if ( is_directory(itr->status()) )
		{
#ifdef linux
			string strLabelPath = itr->path().c_str();
#else
			string strLabelPath = itr->path().string();
#endif

			vector<Mat*> vImages;
			vector<string> vImageNames;

			if(loadImagesOrdered(vImages, vImageNames, iCVLoadMode, strLabelPath, bEqualize))
			{
				if(bAddFlipped)
				{
					addFlippedImages(vImages, vImageNames, 1);
				}

				DataSet* pData = new DataSet();
				pData->setImages(vImages, vImageNames);

#ifdef linux
				string strLabel = itr->path().leaf().c_str();
#else
				string strLabel = itr->path().leaf().string();
#endif

				mCatalogue.insert(pair<string, DataSet*>(strLabel, pData));
			}
		}
	}

	return true;
}

bool mai::IOUtils::loadImagesOrdered(vector<Mat*> &vImages,
		vector<string> &vImageNames,
		int iMode,
		const string &strDirectory,
		bool bEqualize)
{
	boost::filesystem::path directory( boost::filesystem::initial_path<boost::filesystem::path>() );
	directory = boost::filesystem::system_complete( boost::filesystem::path( strDirectory ) );

	if ( !exists( directory ) )
	{
		cout << "[mai::IOUtils::loadImagesOrdered] ERROR loading images. Path does not exist: " << directory << endl;
		return false;
	}
	if(Constants::DEBUG_IMAGE_LOADING) {
		cout << "[mai::IOUtils::loadImagesOrdered] Loading images from " << directory << endl;
	}

	// so we can sort them later
	vector<boost::filesystem::path> catalogueRoot;

	copy(boost::filesystem::directory_iterator(directory), boost::filesystem::directory_iterator(), back_inserter(catalogueRoot));
	sort(catalogueRoot.begin(), catalogueRoot.end());

    for (vector<boost::filesystem::path>::const_iterator itrCR (catalogueRoot.begin()); itrCR != catalogueRoot.end(); ++itrCR)
    {
    	boost::filesystem::path directoryItem = *itrCR;

    	if (!is_directory(directoryItem))
    	{

#ifdef linux
    		string strDirectoryItem = directoryItem.c_str();
#else
    		string strDirectoryItem = directoryItem.string();
#endif

    		loadAndAddImage(vImages,
    				vImageNames,
					iMode,
					strDirectoryItem,
					bEqualize);
    	}
    	else
    	{
    		boost::filesystem::path subDirectory( boost::filesystem::initial_path<boost::filesystem::path>() );
    		subDirectory = boost::filesystem::system_complete(directoryItem);

    		// so we can sort them later
    		vector<boost::filesystem::path> subDirectories;

    		copy(boost::filesystem::directory_iterator(subDirectory), boost::filesystem::directory_iterator(), back_inserter(subDirectories));
    		sort(subDirectories.begin(), subDirectories.end());

    		for (vector<boost::filesystem::path>::const_iterator itrSD (subDirectories.begin()); itrSD != subDirectories.end(); ++itrSD)
    		{
    			boost::filesystem::path subDirectoryItem = *itrSD;

    			if (!is_directory(subDirectoryItem))
    	    	{

#ifdef linux
    				string strDirectoryItem = directoryItem.c_str();
#else
    				string strDirectoryItem = directoryItem.string();
#endif

    				loadAndAddImage(vImages,
    						vImageNames,
							iMode,
							strDirectoryItem,
							bEqualize);
    	    	}
    		}
    	}
    }
	return true;
}

void mai::IOUtils::loadAndAddImage(vector<Mat*> &vImages,
			vector<string> &vImageNames,
			int iMode,
			const string &strDirectoryItem,
			bool bEqualize)
{
	boost::filesystem::path directory( boost::filesystem::initial_path<boost::filesystem::path>() );
	directory = boost::filesystem::system_complete( boost::filesystem::path(strDirectoryItem) );

#ifdef linux
	string strFullFileName = directory.c_str();
	string strFilename = directory.leaf().c_str();
#else
	string strFullFileName = directory.string();
	string strFilename = directory.leaf().string();
#endif

	Mat image;

	if(loadImage(image, iMode, strFullFileName, bEqualize))
	{
		Mat* pImage = new Mat(image);
		vImages.push_back(pImage);
		vImageNames.push_back(strFilename);
	}
}

bool mai::IOUtils::loadImage(Mat &image,
			int iMode,
			const string &strFileName,
			bool bEqualize)
{

	image = imread(strFileName, iMode);

	// Check for valid input
	if(!image.empty())
	{
		if(Constants::DEBUG_IMAGE_LOADING) {
			cout << "[mai::IOUtils::loadImagesOrdered] Image loaded successfully: " << strFileName << endl;
		}

		if (bEqualize)
		{
			if(image.channels() == 3 || image.channels() == 4)
			{
				cvtColor(image, image, CV_BGR2GRAY);
			}

			equalizeHist(image, image);
		}

		return true;
	}
	else
	{
		cout << "[mai::IOUtils::loadImagesOrdered] ERROR loading Image: " << strFileName << endl;
		return false;
	}
}

void mai::IOUtils::addFlippedImages(vector<Mat*> &vImages,
		std::vector<std::string> &vImageNames,
		int iFlipMode)
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

void mai::IOUtils::convertImages(vector<Mat*> &vImages,
		vector<Mat*> &vConvertedImages,
		int iMode )
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

void mai::IOUtils::equalizeImages(vector<Mat*> &vImages,
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

void mai::IOUtils::getMaxImageDimensions(vector<Mat> &vImages,
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

void mai::IOUtils::sampleImage(Mat &image,
		Mat &sampledImage,
		int iHeight,
		int iWidth )
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

void mai::IOUtils::sampleImages(vector<Mat> &vImages,
		vector<Mat> &vSampledImages,
		int iHeight,
		int iWidth )
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

void mai::IOUtils::showImages(vector<Mat> &vImages)
{
	for( Mat image : vImages)
	{
		showImage( image );
	}
}

void mai::IOUtils::showImage(Mat &image)
{
	imshow("Image", image);
	waitKey(0);
}

void mai::IOUtils::showImage(const Mat* image)
{
	imshow("Image", *image);
	waitKey(0);
}

bool mai::IOUtils::createDirectory(const string &strPath)
{
	boost::filesystem::path dir(strPath);
	if (!exists(dir))
	{
		if (!boost::filesystem::create_directory(dir))
		{
			cout << "[mai::IOUtils::CreateDirectory] ERROR creating directory: " << strPath << endl;
			return false;
		}
	}

	return true;
}

void mai::IOUtils::writeImages(vector<Mat*> &vImages,
		vector<string> &vImageNames,
		const string &strPath)
{
	if(!createDirectory(strPath))
	{
		return;
	}

	boost::filesystem::path dir(strPath);
	if (!exists(dir))
	{
		if (!boost::filesystem::create_directory(dir))
		{
			cout << "[mai::IOUtils::writeHOGImages] ERROR creating directory: " << strPath << endl;
		}
	}

	for( unsigned int i = 0; i < vImages.size(); ++i)
	{
		Mat image = *vImages.at(i);
		string strFileNameBase = vImageNames.at(i);

		stringstream sstm;
		sstm << strPath << "/" << strFileNameBase;
		string strFileName = sstm.str();

		imwrite(strFileName, image);
	}
}

void mai::IOUtils::writeHOGImages(mai::DataSet* data,
			const string &strPath,
			const string &strFileNameBase,
			Size imageSize,
			Size cellSize,
			Size blockSize,
			Size blockStride,
			int iNumBins,
			int scaleFactor,
			double vizFactor,
			bool printValue)
{
	if(!createDirectory(strPath))
	{
		return;
	}

	for ( unsigned int i = 0; i < data->getImageCount(); ++i )
	{
		const Mat* image = data->getImageAt(i);
		vector<float> vDescriptorValues;
		data->getDescriptorValuesFromImageAt(i, vDescriptorValues);

		if (vDescriptorValues.size() <= 0 || image == NULL) {
			continue;
		}

		Mat resizedImage, outImage;
		resize(*image, resizedImage, imageSize);
		cvtColor(resizedImage, resizedImage, CV_GRAY2BGR);

		umHOG::getHOGDescriptorVisualImage(outImage,
				resizedImage,
				vDescriptorValues,
				imageSize,
				cellSize,
				blockSize,
				blockStride,
				iNumBins,
				scaleFactor,
				vizFactor,
				printValue);


		stringstream sstm;
		sstm << strPath << "/" << strFileNameBase << "_" << i << ".jpg";
		string strFileName = sstm.str();

		imwrite(strFileName, outImage);
	}
}

void mai::IOUtils::writeSVMs(const map<string, umSVM*> &mSVMs,
			const string &strPath)
{
	if(!createDirectory(strPath))
	{
		return;
	}

	for(map<string, umSVM*>::const_iterator it = mSVMs.begin(); it != mSVMs.end(); it++)
	{
		string strName = it->first;
		umSVM* svm = it->second;

		stringstream sstm;
		sstm << strPath << "/" << strName;
		string strFileName = sstm.str();

		svm->saveSVM(strFileName);
	}
}

void mai::IOUtils::writeMatToCSV(const Mat &data,
				const string &strMatName)
{
	stringstream sstm;
	sstm << strMatName << ".yml";

	FileStorage file(sstm.str(), FileStorage::WRITE);

	file << strMatName << data;

	file.release();
}

bool mai::IOUtils::loadSVMsFromDirectory(map<string, umSVM*> &mSVMs,
		const string &strPath)
{
	boost::filesystem::path directory( boost::filesystem::initial_path<boost::filesystem::path>() );
	directory = boost::filesystem::system_complete( boost::filesystem::path( strPath ) );

	if ( !exists( directory ) )
	{
		cout << "[mai::IOUtils::loadSVMsFromDirectory] ERROR loading svms. Path does not exist: " << directory << endl;
		return false;
	}

	// default construction yields past-the-end
	boost::filesystem::directory_iterator end_itr;

	for ( boost::filesystem::directory_iterator itr( directory ); itr != end_itr; ++itr )
	{
		if ( !is_directory(itr->status()) )
		{
#ifdef linux
			string strFilePath = itr->path().c_str();
			string strLabel = itr->path().stem().c_str();
#else
			string strFilePath = itr->path().string();
			string strLabel = itr->path().stem().string();
#endif

			umSVM* svm = new umSVM();
			svm->loadSVM(strFilePath);

			mSVMs.insert(pair<string, umSVM*>(strLabel, svm));

			if(Constants::DEBUG_SVM_PREDICTION)
			{
				cout << "[mai::IOUtils::loadSVMsFromDirectory] SVM loaded for category " << strLabel << endl;
			}
		}
	}

	return true;
}

