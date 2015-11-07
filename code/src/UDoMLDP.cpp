/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * UDoMLDP.cpp
 *
 *  Created on: Oct 25, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include "UDoMLDP.h"

#include "data/TrainingData.h"
#include "IO/IOUtils.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>


using namespace cv;

using namespace std;


mai::UDoMLDP::UDoMLDP()
{}

mai::UDoMLDP::~UDoMLDP()
{}


void mai::UDoMLDP::UnsupervisedDiscovery(std::string &strFilePathPositives, std::string &strFilePathNegatives)
{
	TrainingData* td = new TrainingData();

	//1. Load data

	std::vector<Mat> images;

	IOUtils::LoadImages ( images, IMREAD_COLOR, strFilePathPositives );

	td->setPositives(images);

	images.clear();

	IOUtils::LoadImages ( images, IMREAD_COLOR, strFilePathNegatives );

	td->setNegatives(images);


	//	1.a discovery dataset D
	//	1.b natural world dataset N
	//2. Split datasets each in 2 equal sized disjoint parts {D1, D2}, {N1, N2}
	//3. Compute HOG descriptors for D1 at multiple resolutions ( at 7 different scales )
	//4. Take random sample patches S from D1 ( ~ 150 per image ) disallowing highly overlapping patches or patches without gradient energy (e.g. sky patches )
	//5. Compute clusters K from S using kmeans with high k = S/4

	Mat data, labels;
	td->getUniformTrainingData(data, labels);

	//6. Loop until convergence, i.e. top patches do not change ( 4 iterations ? ):
	//	6.1 Loop over all K(i) >= 3, skip small clusters
	//		6.1.a train svm on K(i) as positives and N1 as negatives -> classifiers Cnew(i)
	//		6.1.b hard mining ?
	//		6.1.c detect clusters Knew(i) from top m=5 firings for each detector by running Cnew(i) on validation dataset D2 for additional patches to prevent overfitting
	//
	//	6.2 Add Knew to clusters K
	//	6.3 Add Cnew to classifiers C
	//	6.4 Swap datasets D and N
	//7. Loop over all classifiers to compute ranking
	//	7.1 Compute purity: sum up svm detector scores of top r cluster members ( r > m )
	//	7.2 Compute discriminativness: ratio of firings for cluster on D and D u N ( rarely )
	//	7.3 sum up score
	//8. Select top n classifiers from above scores
}
