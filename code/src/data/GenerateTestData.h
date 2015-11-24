/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * GenerateTestData.h
 *
 *  Created on: Oct 11, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_DATA_GENERATETESTDATA_H_
#define SRC_DATA_GENERATETESTDATA_H_

#include <opencv2/core/core.hpp>

namespace mai {

/**
 *
 */
class GenerateTestData {
public:
	GenerateTestData();
	virtual ~GenerateTestData();

	void DrawFilledRectangle( cv::Mat img );
	void DrawFilledCircle( cv::Mat img );

	void GenTestMatrix(cv::Mat data, cv::Mat labels, int iImgSize, int iNumImages);
	void ShowST();

private:
	void AddImageToData(cv::Mat data, cv::Mat image, int iPos);
};

}  // namespace mai

#endif /* SRC_DATA_GENERATETESTDATA_H_ */
