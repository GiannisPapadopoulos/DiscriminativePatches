/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * DataSet.cpp
 *
 *  Created on: Jan 4, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "DataSet.h"

#include <iostream>

#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;

bool mai::DataSet::addDescriptorValuesToImageAt(int iIndex,
		const vector<float> &vDescriptorValues)
{
	if(m_vData[iIndex] != NULL)
	{
		if(m_iDescriptorSize == 0)
		{
			m_iDescriptorSize = vDescriptorValues.size();
		}
		if (m_iDescriptorSize == vDescriptorValues.size())
		{
			m_vData[iIndex]->setDescriptorValues(vDescriptorValues);
			return true;
		}
		else
		{
			cout << "[DataSet::addDescriptorValuesToImageAt] ERROR! Descriptor values have to be of same size : "
					<< m_iDescriptorSize  << "; But index " << iIndex << " has " << vDescriptorValues.size() << endl;
		}
	}
	return false;
}

bool mai::DataSet::addPatchDescriptorValuesToImageAt(int iIndex,
    const std::vector<vector<vector<float>>> &patchDescriptorValues)
{
  if(m_vData[iIndex] != NULL)
  {
    m_vData[iIndex]->setPatchDescriptorValues(patchDescriptorValues);
  }
  return false;
}

int mai::DataSet::setImages(vector<Mat*> &vImages,
		vector<string> &vImageNames)
{
	if(vImages.size() != vImageNames.size())
	{
		cout << "[DataSet::setImages] ERROR! Image and name vector are not of same size!" << endl;
		return 0;
	}

	for(int i = 0; i < vImages.size(); ++i)
	{
		addImage(vImages.at(i), vImageNames.at(i));
	}
	return m_vData.size();
}

void mai::DataSet::removeImages()
{
	for(std::vector<ImageWithDescriptors*>::iterator it = m_vData.begin(); it != m_vData.end(); it++)
	{
		(*it)->removeImage();
	}
}

string mai::DataSet::getFirstSplitBy(const string &strToSplit,
		const string &strDelimiter)
{
	vector<string> strs;
	boost::split(strs, strToSplit, boost::is_any_of(strDelimiter));

	return strs.at(0);
}

void mai::DataSet::getDescriptorsSeparated(int iDivider,
		vector<vector<float> > &vFirstPart,
		vector<vector<float> > &vSecondPart)
{
	int iPercentageValidationImages = m_vData.size()/iDivider > 1 ? m_vData.size()/iDivider : 1;

	vector<ImageWithDescriptors*>::const_iterator it;
	for(it = m_vData.begin(); it != m_vData.begin() + iPercentageValidationImages; it++)
	{
		vFirstPart.push_back((*it)->getDescriptorValues());
	}

	string strDelimiter = "_";

	string strPattern = getFirstSplitBy((*it)->getImageName(), strDelimiter);
	string strMatch = "";

	// step forward if first name identical -> image from same person
	do{
		it++;
		string strMatch = getFirstSplitBy((*it)->getImageName(), strDelimiter);

	} while (strPattern.compare(strMatch) == 0);

	for(; it != m_vData.end(); it++)
	{
		vSecondPart.push_back((*it)->getDescriptorValues());
	}
}

void mai::DataSet::setMaxImageDimensions(Mat* image)
{
	Size s = image->size();
	int iW = s.width;
	int iH = s.height;

	if((m_iMaxHeight != 0 && m_iMaxHeight != iH) || (m_iMaxWidth != 0 && m_iMaxWidth != iW))
	{
		m_bImageSizesUniform = false;
	}
	m_iMaxWidth = std::max(m_iMaxWidth, iW);
	m_iMaxHeight = std::max(m_iMaxHeight, iH);
}
