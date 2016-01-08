/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * faceDetection.h
 *
 *  Created on: Jan 7, 2016
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_UTILS_FACEDETECTION_H_
#define SRC_UTILS_FACEDETECTION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <string>

namespace mai{

class DataSet;

class FaceDetection
{
public:

	virtual ~FaceDetection();

	/**
	 * 	strFilename = "C:/Users/apple/Desktop/Sophia/AI/Project/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml";
	 *
	 */
	static DataSet* detectFaces(DataSet* data,
			const std::string &strFilename,
			double dScale,
			int iMinNeighbors,
			cv::Size minSize,
			cv::Size maxSize);

	static bool detectFace(const cv::Mat &image,
			cv::Mat &face,
			cv::CascadeClassifier cascade,
			double dScale,
			int iMinNeighbors,
			cv::Size minSize,
			cv::Size maxSize);

private:

	FaceDetection();

};

}// namespace mai

#endif /* SRC_UTILS_FACEDETECTION_H_ */
