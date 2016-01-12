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

using namespace std;

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

	/**
	 * Add an image and its filename to the dataset.
	 */
	unsigned int addImage(cv::Mat* image,
			std::string &strImageName)
	{
		m_vData.push_back(new ImageWithDescriptors(image, strImageName));
		setMaxImageDimensions(image);
		return m_vData.size() - 1;
	};

	/**
	 * Get number of images in the dataset.
	 */
	unsigned int getImageCount() const
	{
		return m_vData.size();
	};

	/**
	 * Get the image at position iIndex.
	 * Image may be NULL depending on current state of the processing pipeline !
	 *
	 * @return	image or NULL
	 */
	const cv::Mat* getImageAt(int iIndex) const
	{
		if(m_vData[iIndex] != NULL)
		{
			return m_vData[iIndex]->getImage();
		}
		return NULL;
	};

	/**
	 * Get the name of the image at position iIndex.
	 */
	std::string getImageNameAt(int iIndex) const
	{
		if(m_vData[iIndex] != NULL)
		{
			return m_vData[iIndex]->getImageName();
		}
		return "";
	};

	/**
	 * Add feature vector for image at position iIndex,
	 */
	bool addDescriptorValuesToImageAt(int iIndex,
			const std::vector<float> &vDescriptorValues);

	/**
	 * Add feature vectors for each patch for image at position iIndex
	 */
	bool addPatchDescriptorValuesToImageAt(int iIndex,
			const std::vector<vector<vector<float>>> &patchDescriptorValues);

	/**
	 * Get the feature vector from the image at position iIndex.
	 */
	bool getDescriptorValuesFromImageAt(int iIndex,
			std::vector<float> &vDescriptorValues)
	{
		if(m_vData[iIndex] != NULL)
		{
			vDescriptorValues = m_vData[iIndex]->getDescriptorValues();
			return true;
		}
		return false;
	};

	/**
	 * Get the feature vectors for all patches from the image at position iIndex.
	 */
	bool getPatchDescriptorValuesFromImageAt(int iIndex,
			std::vector<vector<vector<float>>> &vPatchDescriptorValues)
	{
		if(m_vData[iIndex] != NULL)
		{
			vPatchDescriptorValues = m_vData[iIndex]->getPatchDescriptorValues();
			return true;
		}
		return false;
	};

	/**
	 * Add all images with their names from the input vectors.
	 * Uses DataSet::AddImage
	 */
	int setImages(std::vector<cv::Mat*> &vImages,
			std::vector<std::string> &vImageNames);

	/**
	 * Remove the images from the dataset to free memory.
	 * Obviously this should only be done after feature extraction when the images themselves are no longer needed.
	 */
	void removeImages();

	/**
	 * Get maximum dimension of all images in the dataset
	 */
	void getMaxDImensions(int &iMaxWidth, int &iMaxHeight) const
	{
		iMaxHeight = m_iMaxHeight;
		iMaxWidth = m_iMaxWidth;
	};

	/**
	 * Are all the image of the same size ?
	 */
	bool getImageSizesUniform() const
	{
		return m_bImageSizesUniform;
	};

	/**
	 * Get the size of the feature vectors.
	 */
	int getDescriptorValueSize() const
	{
		return m_iDescriptorSize;
	};

	/**
	 * Divide the dataset into two parts according to iDivider.
	 * The division will be adjusted for the first part of the filenames split by '_'
	 * to make sure that images of such name are exclusively contained in first part.
	 */
	void getDescriptorsSeparated(int iDivider,
			std::vector<std::vector<float> > &vFirstPart,
			std::vector<std::vector<float> > &vSecondPart);

private:

	/**
	 * Connects images with its feature vectors.
	 */
	class ImageWithDescriptors
	{
	public:
		ImageWithDescriptors(cv::Mat* pImage,
				std::string strImageName)
	: m_pImage(pImage)
	, m_strImageName(strImageName)
	{};

		virtual ~ImageWithDescriptors()
		{
			removeImage();
		};

		const cv::Mat* getImage() const {
			return m_pImage;
		};

		std::string getImageName() const {
			return m_strImageName;
		};

		const std::vector<float>& getDescriptorValues() const {
			return m_vDescriptorValues;
		};

		void removeImage()
		{
			if(m_pImage != NULL)
			{
				delete m_pImage;
			}

			m_pImage = NULL;
		};

		const std::vector<vector<vector<float>>>& getPatchDescriptorValues() const {
			return m_patchDescriptorValues;
		};

		void setDescriptorValues(const std::vector<float> &vDescriptorValues) {
			m_vDescriptorValues = vDescriptorValues;
		};

		void setPatchDescriptorValues(const std::vector<vector<vector<float>>> &patchDescriptorValues) {
			m_patchDescriptorValues = patchDescriptorValues;
		};

	private:
		cv::Mat*			m_pImage;
		std::vector<float>	m_vDescriptorValues;
		std::vector<vector<vector<float>>> m_patchDescriptorValues;
		std::string			m_strImageName;
	};

	void setMaxImageDimensions(cv::Mat* image);

	std::string getFirstSplitBy(const std::string &strToSplit, const std::string &strDelimiter);

	std::vector<ImageWithDescriptors*>	m_vData;

	int		m_iMaxHeight;
	int		m_iMaxWidth;
	bool	m_bImageSizesUniform;

	int		m_iDescriptorSize;

};

}// namespace mai

#endif /* SRC_DATA_DATASET_H_ */
