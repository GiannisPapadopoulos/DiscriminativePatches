/*
 * umKmeans.cpp
 *
 *  Created on: Jan 11, 2016
 *      Author: giannis
 */

#include "umKmeans.h"

using namespace cv;

using namespace std;

mai::umKmeans::umKmeans() {

}

mai::umKmeans::~umKmeans() {

}

cv::Mat mai::umKmeans::performClustering(cv::Mat& data,
                                         int numClusters,
                                         cv::Mat& labels) {
  cv::Mat centers;
  cv::kmeans(data, numClusters, labels, cv::TermCriteria(), 1,
             cv::KMEANS_RANDOM_CENTERS, centers);
  return centers;
}

