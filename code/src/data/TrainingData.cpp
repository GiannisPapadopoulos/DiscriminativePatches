///*****************************************************************************
// * Master AI project ws15 group 4
// * Mid-level discriminative patches
// *
// * TrainingData.cpp
// *
// *  Created on: Nov 7, 2015
// *      Author: stefan
// *
// *****************************************************************************
// * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
// * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
// *****************************************************************************/

#include "TrainingData.h"

#include "../configuration/Constants.h"

#include <iostream>

using namespace std;

mai::TrainingData::TrainingData(const vector<vector<float> > &vPositiveFeatures,
		const vector<vector<float> > &vNegativeFeatures)
:	m_Data(0, 0, CV_32FC1)
,	m_Labels(0, 0, CV_32SC1)
{
	int iFeatureSize = 0;
	if(vPositiveFeatures.size() > 0 && vNegativeFeatures.size() > 0 && vPositiveFeatures.at(0).size() != vNegativeFeatures.at(0).size())
	{
		cout << "[mai::TrainingData] ERROR! Feature size for positive and negative do not match" << endl;
		return;
	}

	iFeatureSize = vPositiveFeatures.at(0).size();

	if(Constants::DEBUG_DATA_SETUP)
		cout << "[mai::TrainingData] Number of features: " << iFeatureSize << endl;

	int iNumMaxFeatures = vPositiveFeatures.size();
	if(vPositiveFeatures.size() > vNegativeFeatures.size())
	{
		iNumMaxFeatures = vNegativeFeatures.size();
	}

	vector<std::vector<float> > vData;
	vData.insert(end(vData), begin(vPositiveFeatures), begin(vPositiveFeatures) + iNumMaxFeatures);
	vData.insert(end(vData), begin(vNegativeFeatures), begin(vNegativeFeatures) + iNumMaxFeatures);

	// setup training matrices
	unsigned int iNumPatches = vData.size();

	m_Data.create(iNumPatches, iFeatureSize, CV_32FC1);
	m_Labels.create(iNumPatches, 1, CV_32SC1);

	if(Constants::DEBUG_DATA_SETUP)
	{
		cout << "[mai::TrainingData] Training matrix " << m_Data.rows << "x" << m_Data.cols << endl;
		cout << "[mai::TrainingData] Label matrix " << m_Labels.rows << "x" << m_Labels.cols << endl;
	}

	for(unsigned int i = 0; i < iNumPatches ; ++i)
	{
		vector<float> vCurrentData = vData[i];
		for(unsigned int j = 0; j < vCurrentData.size(); ++j)
		{
			m_Data.at<float>(i, j) = vCurrentData[j];
		}

		int iLabel = Constants::SVM_NEGATIVE_LABEL;
		if(i < vPositiveFeatures.size())
		{
			iLabel = Constants::SVM_POSITIVE_LABEL;
		}
		m_Labels.at<int>(i) = iLabel;
	}
}
