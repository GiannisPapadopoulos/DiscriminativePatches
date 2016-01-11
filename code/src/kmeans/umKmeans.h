/*
 * umKmeans.h
 *
 *  Created on: Jan 11, 2016
 *      Author: giannis
 */

#ifndef SRC_KMEANS_UMKMEANS_H_
#define SRC_KMEANS_UMKMEANS_H_

#include <opencv2/core/core.hpp>

namespace mai{

class umKmeans {
 public:
  umKmeans();
  virtual ~umKmeans();

  /**
   * Performs k-means clustering on the input data
   *
   * @param data  The data to cluster, each row is an instance
   * @param numClusters The number of clusters
   * @param labels  The label for each instance
   * @return  The cluster centers
   */
  cv::Mat performClustering(const cv::Mat &data, const int numClusters, const cv::Mat &labels);

};

}// namespace mai

#endif /* KMEANS_UMKMEANS_H_ */
