/*
 * Constants.h
 *
 *  Created on: Nov 25, 2015
 *      Author: giannis
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

class Constants {
public:

	/** Whether to show debug information for loading images */
	static const bool DEBUG_IMAGE_LOADING = false;

	/** Whether to show debug information for dataset separation */
	static const bool DEBUG_DATA_SETUP = false;

	/** Whether to show debug information for the HOG feature extractor */
	static const bool DEBUG_HOG = false;

	/** Whether to show debug information for the PCA feature reduction */
	static const bool DEBUG_PCA = true;

	/** Whether to show debug information for the main algorithm (UDoMLDP) */
	static const bool DEBUG_MAIN_ALG = true;

	/** Whether to show debug information for the svm parts */
	static const bool DEBUG_SVM = false;
	static const bool DEBUG_SVM_PREDICTION = false;
	static const bool DEBUG_MAIN_SVM = false;

	static const bool DEBUG_FACE_DETECTION = false;

	/** svm labels */
	static const int SVM_POSITIVE_LABEL = 1;
	static const int SVM_NEGATIVE_LABEL = 0;

};

#endif /* UTILS_IMAGEDISPLAYUTILS_H_ */
