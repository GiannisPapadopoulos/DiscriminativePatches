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

#include <boost/optional.hpp>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * Application parameters
 */
class Configuration
{
public:
	/**
	 * Loads settings from given configuration file
	 *
	 * @param strFilename	path and name of configuration file
	 */
	Configuration(const std::string &strFilename);

	/**
	 * Nothing to delete here.
	 */
	virtual ~Configuration()
	{};

	/**
	 * OpenCV Documentation says that block size has to be 16x16. Other values are not supported.
	 * Experiments say otherwise !?
	 * Block size has to multiple of cell size.
	 */
	const cv::Size& getBlockSize() const {
		return m_BlockSize;
	};

	/**
	 * 	Block stride has to multiple of cellsize.
	 */
	const cv::Size& getBlockStride() const {
		return m_BlockStride;
	};

	/**
	 * OpenCV Documentation says that cell size has to be 8x8. Other values are not supported.
	 * Experiments say otherwise !?
	 */
	const cv::Size& getCellSize() const {
		return m_CellSize;
	};

	double getHogVizBinScalefactor() const {
		return m_dHOGVizBinScalefactor;
	};

	/**
	 * Linear svm C value defining penalty multiplier for outliers on imperfect separation.
	 */
	double getSvmCValue() const {
		return m_dSVMCValue;
	};

	/**
	 * Divider of dataset size defining validation part, e.g. 4 -> 1/4 of patches will be in validation set.
	 */
	int getDataSetDivider() const {
		return m_iDataSetDivider;
	};

	int getHogVizImageScalefactor() const {
		return m_iHOGVizImageScalefactor;
	};

	/**
	 * Image size has to be multiple of blocksize.
	 * Image will be resized to this size, if the original size is not divideable by cellsize e.g.
	 */
	const cv::Size& getImageSize() const {
		return m_ImageSize;
	};

	/**
	 * Number of gradient orientations for HOG feature extraction.
	 * Optimum is 9.
	 */
	int getNumBins() const {
		return m_iNumBins;
	};

	/**
	 * Path to image catalogue
	 */
	const std::string& getDataFilepath() const {
		return m_strFilepath;
	};

	bool getPredictTrainingData() const {
		return m_bPredictTrainingData;
	};

	bool getWriteSvMs() const {
		return m_bWriteSVMs;
	}

	const std::string& getSvmOutputPath() const {
		return m_strSVMOutputPath;
	}

	/**
	 * Whether to export the hog visualization as image files
	 */
	bool getWriteHogImages() const {
		return m_bWriteHOGImages;
	};

	const std::string& getHogOutputPath() const {
		return m_strHOGOutputPath;
	}

	bool getApplyPCA() const {
		return m_bApplyPCA;
	};

	bool getDetectFaces() const {
		return m_bDetectFaces;
	};

	const std::string getCascadeFilterFileName() const{
		return m_strCascadeFilterFileName;
	}

	double getFDScale() const {
		return m_dFDScale;
	}

	const cv::Size& getFDMaxSize() const {
		return m_FDMaxSize;
	}

	const cv::Size& getFDMinSize() const {
		return m_FDMinSize;
	}

	int getFDMinNeighbors() const {
		return m_iFDMinNeighbors;
	}

private:

	// Helper structs for value conversion
	struct StringToIntTranslator
	{
		typedef std::string internal_type;
		typedef int external_type;

		boost::optional<int> get_value(const std::string &s)
		{
			char *c;
			long l = std::strtol(s.c_str(), &c, 10);
			return boost::make_optional(c != s.c_str(), static_cast<int>(l));
		}
	};

	struct StringToDoubleTranslator
	{
		typedef std::string internal_type;
		typedef double external_type;

		boost::optional<double> get_value(const std::string &s)
		{
			char *c;
			double d = std::strtod(s.c_str(), &c);
			return boost::make_optional(c != s.c_str(), static_cast<double>(d));
		}
	};

	static bool convertStringToBool(std::string str) ;

	struct StringToBoolTranslator
	{
		typedef const std::string internal_type;
		typedef bool external_type;

		boost::optional<bool> get_value(const std::string &s)
		{
			char *c;
			bool b = convertStringToBool(s);
			return boost::make_optional(c != s.c_str(), static_cast<bool>(b));
		}
	};

	// HOG parameters
	cv::Size	m_CellSize;
	cv::Size	m_BlockStride;
	cv::Size	m_BlockSize;

	cv::Size	m_ImageSize;
	int			m_iNumBins;

	// pca parameters
	bool		m_bApplyPCA;

	// svm parameters
	double		m_dSVMCValue;

    bool		m_bPredictTrainingData;

    bool		m_bWriteSVMs;
    std::string	m_strSVMOutputPath;

    // data setup parameters
	std::string	m_strFilepath;
	int			m_iDataSetDivider;

	// HOG output parameters
	bool		m_bWriteHOGImages;
	std::string	m_strHOGOutputPath;
    int			m_iHOGVizImageScalefactor;
    double		m_dHOGVizBinScalefactor;

    // haar cascade parameters
    bool		m_bDetectFaces;
    std::string	m_strCascadeFilterFileName;
    double 		m_dFDScale;
    int			m_iFDMinNeighbors;
    cv::Size	m_FDMinSize;
    cv::Size	m_FDMaxSize;
};

}// namespace mai



#endif /* SRC_CONFIGURATION_H_ */
