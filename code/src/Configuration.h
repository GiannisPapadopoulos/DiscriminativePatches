/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * Configuration.h
 *
 *  Created on: Jan 1, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_CONFIGURATION_H_
#define SRC_CONFIGURATION_H_

#include <string>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * Application parameters
 */
class Configuration
{
public:
	/**
	 * Initializes object
	 */
	Configuration(std::string &strFilename);

	/**
	 * Deletes something
	 */
	virtual ~Configuration()
	{};

	const cv::Size& getBlockSize() const {
		return m_BlockSize;
	}

	const cv::Size& getBlockStride() const {
		return m_BlockStride;
	}

	const cv::Size& getCellSize() const {
		return m_CellSize;
	}

	double getHogVizBinScalefactor() const {
		return m_dHOGVizBinScalefactor;
	}

	double getSvmCValue() const {
		return m_dSVMCValue;
	}

	int getDataSetDivider() const {
		return m_iDataSetDivider;
	}

	int getHogVizImageScalefactor() const {
		return m_iHOGVizImageScalefactor;
	}

	const cv::Size& getImageSize() const {
		return m_ImageSize;
	}

	int getNumBins() const {
		return m_iNumBins;
	}

	const std::string& getDataFilepath() const {
		return m_strFilepath;
	}

private:

	cv::Size	m_CellSize;
	cv::Size	m_BlockStride;
	cv::Size	m_BlockSize;
	cv::Size	m_ImageSize;
	int			m_iNumBins;

	double		m_dSVMCValue;

	std::string	m_strFilepath;
	int			m_iDataSetDivider;

    int			m_iHOGVizImageScalefactor;
    double		m_dHOGVizBinScalefactor;

};

}// namespace mai



#endif /* SRC_CONFIGURATION_H_ */
