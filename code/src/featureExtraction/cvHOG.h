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
	 * Initializes object
	 */
	cvHOG();

	/**
	 * Clears data
	 */
	virtual ~cvHOG();

	/**
	 * Cell and block size have to divide image size.
	 * Cell size has to divide block size.
	 * Block stride has to be multiple of cell size.
	 * descriptorsValues.size = ((img.width / cellsize) - 1) * ((img.height/cellsize) - 1 ) * bins (9) * number of cells per block
	 *
	 * @param[out] descriptorsValues	extracted features
	 * @param[in] image			from which features are extracted
	 * @param blockSize
	 * @param blockStride
	 * @param cellSize
	 * @param winStride		?
	 * @param padding		?
	 */
	static void extractFeatures(std::vector< float> &descriptorsValues,
											cv::Mat &image,
											cv::Size blockSize = cv::Size(16,16),
											cv::Size blockStride = cv::Size(8,8),
											cv::Size cellSize = cv::Size(8,8),
											cv::Size winStride = cv::Size(0,0),
											cv::Size padding = cv::Size(0,0));

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

	//cv::HOGDescriptor	m_pHOG;

};

}// namespace mai

#endif /* SRC_FEATUREEXTRACTION_CVHOG_H_ */
