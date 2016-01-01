/*
 * Constants.h
 *
 *  Created on: Nov 25, 2015
 *      Author: giannis
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_
#endif /* UTILS_IMAGEDISPLAYUTILS_H_ */

class Constants {
 public:

  /** Whether to export the hog visualization as image files */
  static const bool WRITE_HOG_IMAGES = true;

  /** Whether to show debug information for loading images */
    static const bool DEBUG_IMAGE_LOADING = false;
    static const bool DEBUG_DATA_SETUP = false;

    /** Whether to show debug information for the HOG feature extractor */
    static const bool DEBUG_HOG = false;
    static const bool DEBUG_PCA = true;

    /** Whether to show debug information for the main algorithm (UDoMLDP) */
    static const bool DEBUG_MAIN_ALG = false;

    /** Whether to show debug information for the svm */
    static const bool DEBUG_SVM = false;
    static const bool DEBUG_SVM_PREDICTION = false;
   static const bool DEBUG_MAIN_SVM = false;

	// OpenCV Documentation says that blocksize has to be 16x16 and cellsize 8x8. Other values are not supported.
	// Experiments say otherwise !?
	// blockssize and blockstride have to multiples of cellsize
	// image size has to be multiple of blocksize
    static const int HOG_CELLSIZE = 8;
    static const int HOG_BLOCKSTRIDE = 8;
    static const int HOG_BLOCKSIZE = 32;
    static const int HOG_BINS = 9;

	// Image will be resized to this size !
	// If the original size is not divideable by cellsize e.g.
    static const int HOG_IMAGE_SIZE_X = 32;
    static const int HOG_IMAGE_SIZE_Y = 32;

    static const int HOG_VIZ_SCALEFACTOR = 16;
    static constexpr double HOG_VIZ_VIZFACTOR = 2.0;

    /** Reduce featureset */
    static const bool PCA_REDUCTION = false;

    static const int SVM_POSITIVE_LABEL = 1;
    static const int SVM_NEGATIVE_LABEL = 0;

    static constexpr double SVM_C_VALUE = 0.1;

    /** Part of validation data */
    static const int DATESET_DIVIDER = 4;

};
