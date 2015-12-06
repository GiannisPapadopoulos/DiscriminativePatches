/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * umPCA.cpp
 *
 *  Created on: Dec 4, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/


#include "umPCA.h"
#include "../Constants.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

using namespace std;

mai::umPCA::umPCA()
{}
mai::umPCA::~umPCA()
{}

void mai::umPCA::decreaseHOGDescriptorCellsByPCA(vector<float> &inputFeatures,
		vector<float> &reducedFeatures,
		int iNumBins)
{
	int iNumCells = inputFeatures.size() / iNumBins;
	int iCellPosition = 0;
	Mat bins(iNumCells, iNumBins, CV_32FC1);

	for(int i = 0; i < iNumCells; i+= iNumBins)
	{
		for(int j = 0; j < iNumBins; ++j)
		{
			bins.at<float>(i, j) = inputFeatures[iCellPosition];
			iCellPosition++;
		}
	}

	Mat eigenValues, eigenVectors, mean;

	decreaseFeatureSpacebyPCA(bins,
			eigenValues,
			eigenVectors,
			mean);

	for(int i = 0; i < eigenVectors.rows; ++i)
	{
		for(int j = 0; j < eigenVectors.cols; ++j)
		{
			reducedFeatures.push_back(eigenVectors.at<float>(i, j) * eigenValues.at<float>(i));
		}
	}

	if(Constants::DEBUG_PCA) {
		cout << "[mai::umPCA::decreaseHOGDescriptorCellsByPCA] Reduced vectors rows, cols : " << eigenVectors.rows << "," << eigenVectors.cols << endl;
	}
}

void mai::umPCA::decreaseFeatureSpacebyPCA(Mat &features,
									Mat &eigenValues,
									Mat &eigenVectors,
									Mat &mean)
{
	PCA pca(features, Mat(), CV_PCA_DATA_AS_ROW);

	mean = pca.mean.clone();
	eigenValues = pca.eigenvalues.clone();
	eigenVectors = pca.eigenvectors.clone();
}
