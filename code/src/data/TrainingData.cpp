/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * TrainingData.cpp
 *
 *  Created on: Nov 7, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "TrainingData.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <iostream>

using namespace std;


mai::TrainingData::TrainingData()
	: m_iMaxHeight(0)
	, m_iMaxWidth(0)
{}

mai::TrainingData::~TrainingData()
{
	m_vPositives.clear();
	m_vNegatives.clear();
}

std::vector<cv::Mat> mai::TrainingData::getPositives() const
{
	return m_vPositives;
}

void mai::TrainingData::setPositives( std::vector<cv::Mat*> &vImages )
{
	for( cv::Mat* image : vImages)
	{
		m_vPositives.push_back(*image);
		setMaxImageDimensions(*image);
	}
}

std::vector<cv::Mat> mai::TrainingData::getNegatives() const
{
	return m_vNegatives;
}

void mai::TrainingData::setNegatives( std::vector<cv::Mat*> &vImages )
{
	for( cv::Mat* image : vImages)
	{
		m_vNegatives.push_back(*image);
		setMaxImageDimensions(*image);
	}
}

void mai::TrainingData::setUniformImageDimensions( int iHeight, int iWidth )
{
	assert(iHeight > 0 && iWidth > 0);
	m_iMaxHeight = iHeight;
	m_iMaxWidth = iWidth;
}

void mai::TrainingData::getUniformTrainingData( cv::Mat &data, cv::Mat &labels )
{
	int iNumImages = m_vPositives.size() + m_vNegatives.size();

	labels = cv::Mat( iNumImages, 1, CV_32SC1 );
	data = cv::Mat( iNumImages, m_iMaxHeight*m_iMaxWidth, CV_32FC1 );

	for ( unsigned int i = 0; i < m_vPositives.size(); ++i)
	{
		cv::Mat sampledImage;
		sampleImage( m_vPositives[i], sampledImage );
		//data.row(i) = sampledImage.reshape(0, 1);
		addImageToData( data, sampledImage, i );
		labels.at<uchar>(i) = 1;
	}

	for ( unsigned int i = 0; i < m_vNegatives.size(); ++i)
	{
		cv::Mat sampledImage;
		sampleImage( m_vNegatives[i], sampledImage );
		addImageToData( data, sampledImage, i );
		labels.at<uchar>(i) = -1;
	}

	cout << "[mai::TrainingData::getUniformTrainingData] Sampled " << iNumImages << " images to " << m_iMaxHeight << "x" << m_iMaxWidth << endl;
}

void mai::TrainingData::addImageToData( cv::Mat data, cv::Mat image, int iPos)
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

void mai::TrainingData::sampleImage( cv::Mat &image, cv::Mat &sampledImage )
{
	cv::Size s = image.size();
	int iW = s.width;
	int iH = s.height;

	if ( iW * iH == m_iMaxHeight * m_iMaxWidth )
		sampledImage = image;
	else
	{
		if ( iW * iH < m_iMaxHeight * m_iMaxWidth )
		{
			cv::pyrUp( image, sampledImage, cv::Size( m_iMaxWidth, m_iMaxHeight ) );
		}
		else
		{
			cv::pyrDown( image, sampledImage, cv::Size( m_iMaxWidth, m_iMaxHeight ) );
		}
	}

	assert( !sampledImage.empty() );
}

void mai::TrainingData::setMaxImageDimensions( cv::Mat image )
{
	cv::Size s = image.size();
	int iW = s.width;
	int iH = s.height;

	m_iMaxWidth = std::max(m_iMaxWidth, iW);
	m_iMaxHeight = std::max(m_iMaxHeight, iH);
}
