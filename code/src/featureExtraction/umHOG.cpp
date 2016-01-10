/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * cvHOG.cpp
 *
 *  Created on: Nov 18, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/


#include "umHOG.h"
#include "../data/DataSet.h"
#include "../Constants.h"
#include "umPCA.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

#include <cmath>

using namespace cv;

using namespace std;

mai::umHOG::umHOG()
{}
mai::umHOG::~umHOG()
{}

void mai::umHOG::computeHOGForDataSet(DataSet* data,
		Size imageSize,
		Size blockSize,
		Size blockStride,
		Size cellSize,
		int iNumBins,
		Size winStride,
		Size padding,
		bool bApplyPCA)
{
	// Extract features from all images in dataset.
	for(unsigned int i = 0; i < data->getImageCount(); ++i)
	{
		const Mat* image = data->getImageAt(i);

		vector<float> descriptorsValues;

		umHOG::extractFeatures(descriptorsValues,
				*image,
				imageSize,
				blockSize,
				blockStride,
				cellSize,
				iNumBins,
				winStride,
				padding,
				bApplyPCA);

		data->addDescriptorValuesToImageAt(i, descriptorsValues);
	}
}

void mai::umHOG::extractFeatures(vector<float> &descriptorsValues,
								const Mat &image,
								Size imageSize,
								Size blockSize,
								Size blockStride,
								Size cellSize,
								int iNumBins,
								Size winStride,
								Size padding,
								bool bApplyPCA)
{
	if(Constants::DEBUG_HOG) {
	  cout << "[mai::cvHOG::computeHOGForDataSet] resizing image to " << imageSize << endl;
	}
	Mat resizedImage;
	resize(image, resizedImage, imageSize);

	if(Constants::DEBUG_HOG) {
		cout << "[mai::cvHOG::extractFeatures] computing HOG with blocksize " << blockSize
				<< ", blockstride " << blockStride << ", cellSize " << cellSize << ", num Bins " << iNumBins << endl;
	}

	HOGDescriptor hog( imageSize, blockSize, blockStride, cellSize, iNumBins);

	vector< Point> locations;

	if(bApplyPCA)
	{
		vector<float> unreducedFeatures;
		hog.compute( resizedImage, unreducedFeatures, winStride, padding, locations);

		umPCA::decreaseHOGDescriptorCellsByPCA(
				unreducedFeatures,
				descriptorsValues,
				iNumBins);

		if(Constants::DEBUG_HOG) {
			cout << "[mai::cvHOG::extractFeatures] Applied PCA reduction, original feature size: " << unreducedFeatures.size() << endl;
		}
	}
	else
	{
		hog.compute( resizedImage, descriptorsValues, winStride, padding, locations);
	}

	if(Constants::DEBUG_HOG) {
		cout << "[mai::cvHOG::extractFeatures] descriptor size: " << descriptorsValues.size() << endl;
	}
}

void mai::umHOG::getHOGDescriptorVisualImage(Mat &outImage,
							   Mat &origImg,
                               vector<float> &descriptorValues,
                               Size winSize,
                               Size cellSize,
							   Size blockSize,
							   Size blockStride,
							   int iNumBins,
                               int scaleFactor,
                               double vizFactor,
							   bool printValue)
{
	resize(origImg, outImage, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));

	// dividing 180° into iNumBins bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14/(float)iNumBins;

	// prepare data structure: iNumBins orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
	int cells_in_y_dir = winSize.height / cellSize.height;
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
	int blocks_in_x_dir = (winSize.width / blockStride.width) - (blockSize.width / blockStride.width - 1);
	int blocks_in_y_dir = (winSize.height / blockStride.height) - (blockSize.height / blockStride.height - 1);
	int iNumCellsPerBlockX = blockSize.width / cellSize.width;
	int iNumCellsPerBlockY = blockSize.height / cellSize.height;
	int iNumCellsPerBlock = iNumCellsPerBlockX * iNumCellsPerBlockY;

	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter   = new int*[cells_in_y_dir];
	for (int y=0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x=0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[iNumBins];
			cellUpdateCounter[y][x] = 0;

			for (int bin=0; bin<iNumBins; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
		{
			for (int cellNr=0; cellNr<iNumCellsPerBlock; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx + cellNr / iNumCellsPerBlockX;
				int celly = blocky + cellNr % iNumCellsPerBlockY;

				for (int bin=0; bin<iNumBins; bin++)
				{
					float gradientStrength = descriptorValues[ descriptorDataIdx ];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;
				} // for (all bins)

					// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)
		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (int celly=0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx=0; cellx<cells_in_x_dir; cellx++)
		{
			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin=0; bin<iNumBins && NrUpdatesForThisCell != 0; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (int celly=0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx=0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width/2;
			int my = drawY + cellSize.height/2;

			rectangle(outImage,
					Point(drawX*scaleFactor,drawY*scaleFactor),
					Point((drawX+cellSize.width)*scaleFactor,
							(drawY+cellSize.height)*scaleFactor),
							CV_RGB(100,100,100),
							1);

			float cellGradient = 0;
			float firstBin = 0;

			// draw in each cell all gradient strengths
			for (int bin=0; bin<iNumBins; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength==0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

				float dirVecX = cos( currRad );
				float dirVecY = sin( currRad );
				float maxVecLen = cellSize.width/2;
				float scale = vizFactor; // just a visual_imagealization scale,
				// to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(outImage,
						Point(x1*scaleFactor,y1*scaleFactor),
						Point(x2*scaleFactor,y2*scaleFactor),
						CV_RGB(0,255,0),
						1);

				cellGradient += currentGradStrength;

				if(bin == 0)
					firstBin = currentGradStrength;

			} // for (all bins)

			if(printValue)
			{
				cellGradient /= iNumBins;
				cellGradient = round( cellGradient * 1000.0 ) / 1000.0;

				string text = boost::lexical_cast<std::string>(cellGradient);
				//string text = boost::lexical_cast<std::string>(firstBin);
				int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
				double fontScale = 0.5;
				int thickness = 1;
				int baseline=0;
				Size textSize = getTextSize(text, fontFace,
						fontScale, thickness, &baseline);
				// then put the text itself
				putText(outImage, text, Point(drawX*scaleFactor,my*scaleFactor), fontFace, fontScale,
						Scalar::all(255), thickness, 8);
			}

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y=0; y<cells_in_y_dir; y++)
	{
		for (int x=0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;
}
