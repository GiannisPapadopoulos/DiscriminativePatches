/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * cvHOG.h
 *
 *  Created on: Nov 18, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_FEATUREEXTRACTION_CVHOG_H_
#define SRC_FEATUREEXTRACTION_CVHOG_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * Input and output tools
 */
class cvHOG
{
public:

	/**
	 * Clears data
	 */
	virtual ~cvHOG();

	/**
	 * Cell and block size have to divide image size.
	 * Cell size has to divide block size.
	 * descriptorsValues.size = ((img.width / cellsize) - 1) * ((img.height/cellsize) - 1 ) * bins (9) * number of cells per block
	 *
	 * @param[out] descriptorsValues	extracted features
	 * @param[in] image			from which features are extracted
	 * @param cellSize
	 * @param blockStride
	 * @param blockSize
	 */
	static void extractFeatures(std::vector< float> &descriptorsValues,
											cv::Mat &image,
											cv::Size cellSize,
											cv::Size blockStride,
											cv::Size blockSize);

	/**
	 * Visualize HOG descriptor on image
	 * @see http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization#computing_the_hog_descriptor_using_opencv
	 */
	static void getHOGDescriptorVisualImage(cv::Mat &outImage,
									   cv::Mat &origImg,
	                                   std::vector<float> &descriptorValues,
	                                   cv::Size winSize,
	                                   cv::Size cellSize,
	                                   int scaleFactor,
	                                   double vizFactor);

private:
	/**
	 * Initializes object
	 */
	cvHOG();

};

}// namespace mai

#endif /* SRC_FEATUREEXTRACTION_CVHOG_H_ */
