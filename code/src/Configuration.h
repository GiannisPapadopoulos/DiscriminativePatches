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
	};

	const cv::Size& getBlockStride() const {
		return m_BlockStride;
	};

	const cv::Size& getCellSize() const {
		return m_CellSize;
	};

	double getHogVizBinScalefactor() const {
		return m_dHOGVizBinScalefactor;
	};

	double getSvmCValue() const {
		return m_dSVMCValue;
	};

	int getDataSetDivider() const {
		return m_iDataSetDivider;
	};

	int getHogVizImageScalefactor() const {
		return m_iHOGVizImageScalefactor;
	};

	const cv::Size& getImageSize() const {
		return m_ImageSize;
	};

	int getNumBins() const {
		return m_iNumBins;
	};

	const std::string& getDataFilepath() const {
		return m_strFilepath;
	};

	bool getPredictTrainingData() const {
		return m_bPredictTrainingData;
	};

	bool getWriteHogImages() const {
		return m_bWriteHOGImages;
	};

	bool getApplyPCA() const {
		return m_bApplyPCA;
	};


private:

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

	cv::Size	m_CellSize;
	cv::Size	m_BlockStride;
	cv::Size	m_BlockSize;
	cv::Size	m_ImageSize;
	int			m_iNumBins;

	bool		m_bApplyPCA;

	double		m_dSVMCValue;

	std::string	m_strFilepath;
	int			m_iDataSetDivider;

	bool		m_bWriteHOGImages;
    int			m_iHOGVizImageScalefactor;
    double		m_dHOGVizBinScalefactor;

    bool		m_bPredictTrainingData;

};

}// namespace mai



#endif /* SRC_CONFIGURATION_H_ */
