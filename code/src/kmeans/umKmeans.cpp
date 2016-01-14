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

cv::Mat mai::umKmeans::performClustering(const cv::Mat& data,
                                         const int numClusters,
                                         const cv::Mat& labels) {
  cv::kmeans(data, numClusters, labels, cv::TermCriteria(), 1,
             cv::KMEANS_RANDOM_CENTERS, m_clusterCenters);
  return m_clusterCenters;
}

