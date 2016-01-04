/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * DataSet.h
 *
 *  Created on: Nov 23, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_DATA_DATASET_H_
#define SRC_DATA_DATASET_H_

#include <opencv2/core/core.hpp>

#include <vector>
#include <iostream>

namespace mai{

/**
 * Dataset with images and corresponding feature vectors
 */
class DataSet
{
public:
	/**
	 * Initializes object
	 */
	DataSet()
		: m_iMaxHeight(0)
		, m_iMaxWidth(0)
		, m_bImageSizesUniform(true)
		, m_iDescriptorSize(0)
	{};

	/**
	 * Clears data
	 */
	virtual ~DataSet()
	{
		for(ImageWithDescriptors* object : m_vData)
		{
			delete object;
		}
		m_vData.clear();
	};

	unsigned int addImage(cv::Mat* image)
	{
		m_vData.push_back(new ImageWithDescriptors(image));
		setMaxImageDimensions(image);
		return m_vData.size() - 1;
	};

	unsigned int getImageCount() const
	{
		return m_vData.size();
	};

	const cv::Mat* getImageAt(int iIndex) const
	{
		if(m_vData[iIndex] != NULL)
		{
			return m_vData[iIndex]->getImage();
		}
		return NULL;
	};

	bool addDescriptorValuesToImageAt(int iIndex, const std::vector<float> &vDescriptorValues)
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
				std::cout << "[DataSet::addDescriptorValuesToImageAt] ERROR! Descriptor values have to be of same size : "
						<< m_iDescriptorSize  << "; But index " << iIndex << " has " << vDescriptorValues.size() << std::endl;
			}
		}
		return false;
	};

	bool getDescriptorValuesFromImageAt(int iIndex, std::vector<float> &vDescriptorValues)
	{
		if(m_vData[iIndex] != NULL)
		{
			vDescriptorValues = m_vData[iIndex]->getDescriptorValues();
			return true;
		}
		return false;
	};

	int setImages(std::vector<cv::Mat*> &vImages)
	{
		for( cv::Mat* image : vImages)
		{
			addImage(image);
		}
		return m_vData.size();
	};

	void getMaxDImensions(int &iMaxWidth, int &iMaxHeight) const
	{
		iMaxHeight = m_iMaxHeight;
		iMaxWidth = m_iMaxWidth;
	};

	bool getImageSizesUniform() const
	{
		return m_bImageSizesUniform;
	};

	int getDescriptorValueSize() const
	{
		return m_iDescriptorSize;
	};

	void getDescriptorsSeparated(int iDivider, std::vector<std::vector<float> > &vFirstPart, std::vector<std::vector<float> > &vSecondPart)
	{
		int iPercentageValidationImages = m_vData.size()/iDivider > 1 ? m_vData.size()/iDivider : 1;

		for(std::vector<ImageWithDescriptors*>::const_iterator it = m_vData.begin(); it != m_vData.begin() + iPercentageValidationImages; it++)
		{
			vFirstPart.push_back((*it)->getDescriptorValues());
		}

		for(std::vector<ImageWithDescriptors*>::const_iterator it = m_vData.begin() + iPercentageValidationImages; it != m_vData.end(); it++)
		{
			vSecondPart.push_back((*it)->getDescriptorValues());
		}
	};

private:

	/**
	 * Connects images with its feature vectors.
	 */
	class ImageWithDescriptors
	{
	public:
		ImageWithDescriptors(cv::Mat* pImage)
			: m_pImage(pImage)
		{};

		virtual ~ImageWithDescriptors()
		{
			delete m_pImage;
		};

		const cv::Mat* getImage() const {
			return m_pImage;
		};

		void setImage(cv::Mat* pImage) {
			m_pImage = pImage;
		};

		const std::vector<float>& getDescriptorValues() const {
			return m_vDescriptorValues;
		};

		void setDescriptorValues(const std::vector<float> &vDescriptorValues) {
			m_vDescriptorValues = vDescriptorValues;
		};

	private:
		cv::Mat*			m_pImage;
		std::vector<float>	m_vDescriptorValues;
	};

	void setMaxImageDimensions(cv::Mat* image)
	{
		cv::Size s = image->size();
		int iW = s.width;
		int iH = s.height;

		if((m_iMaxHeight != 0 && m_iMaxHeight != iH) || (m_iMaxWidth != 0 && m_iMaxWidth != iW))
		{
			m_bImageSizesUniform = false;
		}
		m_iMaxWidth = std::max(m_iMaxWidth, iW);
		m_iMaxHeight = std::max(m_iMaxHeight, iH);
	};

	std::vector<ImageWithDescriptors*>	m_vData;

	int		m_iMaxHeight;
	int		m_iMaxWidth;
	bool	m_bImageSizesUniform;

	int		m_iDescriptorSize;

};

}// namespace mai

#endif /* SRC_DATA_DATASET_H_ */
