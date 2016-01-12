/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * Configuration.cpp
 *
 *  Created on: Jan 1, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "Configuration.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace mai;

mai::Configuration::Configuration(const string &strFilename)
{
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(strFilename, pt);

	StringToIntTranslator trInt;
	StringToDoubleTranslator trDouble;
	StringToBoolTranslator trBool;

	m_CellSize = Size(pt.get<int>("HOG.CELLSIZE_X", 8, trInt), pt.get<int>("HOG.CELLSIZE_Y", 8, trInt));
	m_BlockStride = Size(pt.get<int>("HOG.BLOCKSTRIDE_X", 8, trInt), pt.get<int>("HOG.BLOCKSTRIDE_Y", 8, trInt));
	m_BlockSize = Size(pt.get<int>("HOG.BLOCKSIZE_X", 32, trInt), pt.get<int>("HOG.BLOCKSIZE_Y", 32, trInt));
	m_ImageSize = Size(pt.get<int>("HOG.IMAGE_SIZE_X", 32, trInt), pt.get<int>("HOG.IMAGE_SIZE_Y", 32, trInt));
	m_iNumBins = pt.get<int>("HOG.BINS", 9, trInt);

	m_bApplyPCA = pt.get<bool>("HOG.APPLY_PCA", false, trBool);

	m_bWriteHOGImages = pt.get<bool>("HOG.WRITE_HOGIMAGES", true, trBool);
	m_strHOGOutputPath = pt.get<std::string>("HOG.FILEPATH", "outHOG");
	m_iHOGVizImageScalefactor = pt.get<int>("HOG.VIZ_IMAGE_SCALEFACTOR", 4, trInt);
	m_dHOGVizBinScalefactor = pt.get<double>("HOG.VIZ_BIN_SCALEFACTOR", 2.0, trDouble);

	m_iDataSetDivider = pt.get<int>("DATA.DATESET_DIVIDER", 4, trInt);
	m_strFilepath = pt.get<std::string>("DATA.FILEPATH");
	m_bAddFlippedImages = pt.get<bool>("DATA.ADD_FLIPPED_IMAGES", true, trBool);

	m_dSVMCValue = pt.get<double>("SVM.C_VALUE", 0.1, trDouble);
	m_bWriteSVMs = pt.get<bool>("SVM.WRITE_SVMS", true, trBool);
	m_strSVMOutputPath = pt.get<std::string>("SVM.FILEPATH", "outSVM");

	m_bPredictTrainingData = pt.get<bool>("SVM.PREDICT_TRAININGDATA", true, trBool);

	m_bDetectFaces = pt.get<bool>("FACE_DETECTION.DETECT_FACES", false, trBool);
	m_strCascadeFilterFileName = pt.get<std::string>("FACE_DETECTION.FILENAME", ".");
	m_dFDScale = pt.get<double>("FACE_DETECTION.SCALE", 1.1, trDouble);
	m_iFDMinNeighbors = pt.get<int>("FACE_DETECTION.MIN_NEIGHBORS", 3, trInt);
	m_FDMinSize = Size(pt.get<int>("FACE_DETECTION.MIN_SIZE", 240, trInt), pt.get<int>("FACE_DETECTION.MIN_SIZE", 240, trInt));
	m_FDMaxSize = Size(pt.get<int>("FACE_DETECTION.MAX_SIZE", 480, trInt), pt.get<int>("FACE_DETECTION.MAX_SIZE",480, trInt));

	m_bPerformClustering = pt.get<bool>("CLUSTERING.PERFORM_CLUSTERING", false, trBool);
}

bool mai::Configuration::convertStringToBool(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	std::istringstream is(str);
	bool b;
	is >> std::boolalpha >> b;
	return b;
};
