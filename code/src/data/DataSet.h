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

namespace mai{

/**
 * Dataset with images and features
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

	int addImage(cv::Mat* image)
	{
		m_vData.push_back(new ImageWithDescriptors(image));
		setMaxImageDimensions(image);
		return m_vData.size() - 1;
	};

	int getImageCount() const
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
			m_vData[iIndex]->setDescriptorValues(vDescriptorValues);
			return true;
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

	void getMaxDImensions(int &iMaxWidth, int &iMaxHeight)
	{
		iMaxHeight = m_iMaxHeight;
		iMaxWidth = m_iMaxWidth;
	};

	int getDescriptorValueSize() const
	{
		return m_iDescriptorSize;
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

		m_iMaxWidth = std::max(m_iMaxWidth, iW);
		m_iMaxHeight = std::max(m_iMaxHeight, iH);
	};

	std::vector<ImageWithDescriptors*>	m_vData;

	int	m_iMaxHeight;
	int	m_iMaxWidth;

	int m_iDescriptorSize;

};

}// namespace mai

#endif /* SRC_DATA_DATASET_H_ */
