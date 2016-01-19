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

#ifndef SRC_CONFIGURATION_CONFIGURATION_H_
#define SRC_CONFIGURATION_CONFIGURATION_H_

#include <string>

#include <boost/optional.hpp>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * Adjustable application parameters.
 * The values are defined in a configuration file with ini-format.
 * The file is loaded and evaluated on construction.
 */
class Configuration
{
public:

	/**
	 * Loads and evaluates settings from given configuration file
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
	 * Application operating modes encoded.
	 */
	enum appMode
	{
		Undef = 0,
		Train = 1,
		Retrain,
		Predict
	};

	/**
	 * @return	Application operating mode: TRAIN, RETRAIN or PREDICT.
	 */
	const appMode getApplicationMode() const {
		return m_AppMode;
	};

	/**
	 * @return	Filepath to input data for prediction on trained classifiers.
	 */
	const std::string& getImageInputPath() const {
		return m_strImageInputPath;
	};

	/**
	 * @return	Fielpath to trained svms for classification or retraining.
	 */
	const std::string& getSVMInputPath() const {
		return m_strSVMInputPath;
	};

	/**
	 * OpenCV Documentation says that block size has to be 16x16. Other values are not supported.
	 * Experiments say otherwise !?
	 * Block size has to multiple of cell size.
	 *
	 * @return	HOG feature extraction block size.
	 */
	const cv::Size& getBlockSize() const {
		return m_BlockSize;
	};

	/**
	 * 	Block stride has to multiple of cellsize.
	 *
	 * 	@return	HOG feature extraction block stride.
	 */
	const cv::Size& getBlockStride() const {
		return m_BlockStride;
	};

	/**
	 * OpenCV Documentation says that cell size has to be 8x8. Other values are not supported.
	 * Experiments say otherwise !?
	 *
	 * @return	HOG feature extraction cell size.
	 */
	const cv::Size& getCellSize() const {
		return m_CellSize;
	};

	/**
	 * @return	HOG visualization scaling factor for gradient visualization.
	 */
	double getHogVizBinScalefactor() const {
		return m_dHOGVizBinScalefactor;
	};

	/**
	 * @return	Linear svm C value defining penalty multiplier for outliers on imperfect separation.
	 */
	double getSvmCValue() const {
		return m_dSVMCValue;
	};

	/**
	 * @return	Divider of dataset size defining validation part, e.g. 2 -> 1/2 of patches will be in validation set.
	 */
	int getDataSetDivider() const {
		return m_iDataSetDivider;
	};

	/**
	 * @return	HOG visualization scaling factor for the image.
	 */
	int getHogVizImageScalefactor() const {
		return m_iHOGVizImageScalefactor;
	};

	/**
	 * Image size has to be multiple of blocksize.
	 * Image will be resized to this size, if the original size is not divideable by cellsize e.g.
	 * and to make sure, all images are of same size.
	 *
	 * @return	HOG feature extraction image size.
	 */
	const cv::Size& getImageSize() const {
		return m_ImageSize;
	};

	/**
	 * Number of gradient orientations for HOG feature extraction.
	 * Optimum is 9.
	 *
	 * @return	HOG feature extraction number of bins.
	 */
	int getNumBins() const {
		return m_iNumBins;
	};

	/**
	 * @return	Path to image cataloge used for training the classifiers.
	 */
	const std::string& getDataFilepath() const {
		return m_strFilepath;
	};

	bool getCrossValidate() const {
		return m_bCrossValidate;
	};

	/**
	 * @return	Should training data be predicted by the trained classifiers ?
	 */
	bool getPredictTrainingData() const {
		return m_bPredictTrainingData;
	};

	/**
	 * @return	Should the trained classifiers be saved to disc ?
	 */
	bool getWriteSvMs() const {
		return m_bWriteSVMs;
	};

	/**
	 * @return	Output filepath to save trained classifiers, if desired. @see getWriteSvMs.
	 */
	const std::string& getSvmOutputPath() const {
		return m_strSVMOutputPath;
	};

	/**
	 * @return	Whether to export the hog visualization as image files.
	 */
	bool getWriteHogImages() const {
		return m_bWriteHOGImages;
	};

	/**
	 * @return	Output filepath to save HOG visualization images, if desired. @see getWriteHogImages.
	 */
	const std::string& getHogOutputPath() const {
		return m_strHOGOutputPath;
	};

	/**
	 * @return	Should the HOG features be reduced applying Principal Component Analysis ?
	 */
	bool getApplyPCA() const {
		return m_bApplyPCA;
	};

	/**
	 * @return	Should images be processed by Haar Cascade Filter to detect faces ?
	 */
	bool getDetectFaces() const {
		return m_bDetectFaces;
	};

	/**
	 * @return	Path and filename of trained Haar Cascade Classifier.
	 */
	const std::string getCascadeFilterFileName() const{
		return m_strCascadeFilterFileName;
	};

	/**
	 * @return	Scaling factor for Haar Cascade Filter.
	 */
	double getFDScale() const {
		return m_dFDScale;
	};

	/**
	 * @return	Maximum face size detectable by Haar Cascade Filter.
	 */
	const cv::Size& getFDMaxSize() const {
		return m_FDMaxSize;
	};

	/**
	 * @return	Minimum face size detectable by Haar Cascade Filter.
	 */
	const cv::Size& getFDMinSize() const {
		return m_FDMinSize;
	};

	/**
	 * @return	Minimum number of ?
	 */
	int getFDMinNeighbors() const {
		return m_iFDMinNeighbors;
	};

	/**
	 * @return	Should the clustering be part of the processing pipeline ?
	 */
	bool getPerfromClustering() const {
		return m_bPerformClustering;
	};
	
	/**
	 * @return	Should the number of training samples be enlarged by adding a horizontally flipped version of each image ?
	 */
	bool getAddFlipedImages() const {
		return m_bAddFlippedImages;
	};

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


	// Main application parameters
	appMode		m_AppMode;
	std::string	m_strImageInputPath;
	std::string	m_strSVMInputPath;

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
	bool		m_bCrossValidate;
    bool		m_bPredictTrainingData;

    bool		m_bWriteSVMs;
    std::string	m_strSVMOutputPath;

    // data setup parameters
	std::string	m_strFilepath;
	int			m_iDataSetDivider;
	bool		m_bAddFlippedImages;

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

    // k-means parameters
    bool		m_bPerformClustering;
};

}// namespace mai



#endif /* SRC_CONFIGURATION_CONFIGURATION_H_ */
