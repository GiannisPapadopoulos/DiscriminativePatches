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
	/**
	 * 	strFilename = "C:/Users/apple/Desktop/Sophia/AI/Project/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml";
	 *
	 */
	FaceDetection(const std::string &strFilename);

	virtual ~FaceDetection();

	DataSet* detectFaces(DataSet* data);

	void detectFace(const cv::Mat &image,
			cv::Mat &face);

private:

	CvHaarClassifierCascade* m_Cascade;
};

}// namespace mai

#endif /* SRC_UTILS_FACEDETECTION_H_ */
