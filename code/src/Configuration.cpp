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


mai::Configuration::Configuration(const string &strFilename)
{
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(strFilename, pt);

	StringToIntTranslator trInt;
	StringToDoubleTranslator trDouble;
	StringToBoolTranslator trBool;

	m_CellSize = Size(pt.get<int>("HOG.CELLSIZE_X", trInt), pt.get<int>("HOG.CELLSIZE_Y", trInt));
	m_BlockStride = Size(pt.get<int>("HOG.BLOCKSTRIDE_X", trInt), pt.get<int>("HOG.BLOCKSTRIDE_Y", trInt));
	m_BlockSize = Size(pt.get<int>("HOG.BLOCKSIZE_X", trInt), pt.get<int>("HOG.BLOCKSIZE_Y", trInt));
	m_ImageSize = Size(pt.get<int>("HOG.IMAGE_SIZE_X", trInt), pt.get<int>("HOG.IMAGE_SIZE_Y", trInt));
	m_iNumBins = pt.get<int>("HOG.BINS", trInt);

	m_bApplyPCA = pt.get<bool>("HOG.APPLY_PCA", trBool);

	m_bWriteHOGImages = pt.get<bool>("HOG.WRITE_HOGIMAGES", trBool);
	m_iHOGVizImageScalefactor = pt.get<int>("HOG.VIZ_IMAGE_SCALEFACTOR", trInt);
	m_dHOGVizBinScalefactor = pt.get<double>("HOG.VIZ_BIN_SCALEFACTOR", trDouble);

	m_iDataSetDivider = pt.get<int>("DATA.DATESET_DIVIDER", trInt);
	m_strFilepath = pt.get<std::string>("DATA.FILEPATH");

	m_dSVMCValue = pt.get<double>("SVM.C_VALUE", trDouble);

	m_bPredictTrainingData = pt.get<bool>("SVM.PREDICT_TRAININGDATA", trBool);

	m_bDetectFaces = pt.get<bool>("FACE_DETECTION.DETECT_FACES", trBool);
	m_strCascadeFilterFileName = pt.get<std::string>("FACE_DETECTION.FILENAME");
	m_dFDScale = pt.get<double>("FACE_DETECTION.SCALE", trDouble);
	m_iFDMinNeighbors = pt.get<int>("FACE_DETECTION.MIN_NEIGHBORS", trInt);
	m_FDMinSize = Size(pt.get<int>("FACE_DETECTION.MIN_SIZE", trInt), pt.get<int>("FACE_DETECTION.MIN_SIZE", trInt));
	m_FDMaxSize = Size(pt.get<int>("FACE_DETECTION.MAX_SIZE", trInt), pt.get<int>("FACE_DETECTION.MAX_SIZE", trInt));
}

bool mai::Configuration::convertStringToBool(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	std::istringstream is(str);
	bool b;
	is >> std::boolalpha >> b;
	return b;
};
