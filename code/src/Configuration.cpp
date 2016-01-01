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
#include <boost/optional.hpp>

#include <iostream>

using namespace std;
using namespace cv;

struct string_to_int_translator
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

struct string_to_double_translator
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

mai::Configuration::Configuration(string &strFilename)
{
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini("config.ini", pt);

	string_to_int_translator trInt;
	string_to_double_translator trDouble;

	m_CellSize = Size(pt.get<int>("HOG.CELLSIZE_X", trInt), pt.get<int>("HOG.CELLSIZE_Y", trInt));
	m_BlockStride = Size(pt.get<int>("HOG.BLOCKSTRIDE_X", trInt), pt.get<int>("HOG.BLOCKSTRIDE_Y", trInt));
	m_BlockSize = Size(pt.get<int>("HOG.BLOCKSIZE_X", trInt), pt.get<int>("HOG.BLOCKSIZE_Y", trInt));
	m_ImageSize = Size(pt.get<int>("HOG.IMAGE_SIZE_X", trInt), pt.get<int>("HOG.IMAGE_SIZE_Y", trInt));
	m_iNumBins = pt.get<int>("HOG.BINS", trInt);

	m_iHOGVizImageScalefactor = pt.get<int>("HOG.VIZ_IMAGE_SCALEFACTOR", trInt);
	m_dHOGVizBinScalefactor = pt.get<double>("HOG.VIZ_BIN_SCALE_FACTOR", trDouble);

	m_iDataSetDivider = pt.get<int>("DATA.DATESET_DIVIDER", trInt);
	m_strFilepath = pt.get<std::string>("DATA.FILEPATH");

	m_dSVMCValue = pt.get<double>("SVM.C_VALUE", trDouble);

}
