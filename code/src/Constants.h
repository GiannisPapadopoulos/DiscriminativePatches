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
  static const bool WRITE_HOG_IMAGES = false;

  /** Whether to show debug information for loading images */
    static const bool DEBUG_IMAGE_LOADING = false;

    /** Whether to show debug information for the HOG feature extractor */
    static const bool DEBUG_HOG = false;
    static const bool DEBUG_PCA = true;

    /** Whether to show debug information for the main algorithm (UDoMLDP) */
    static const bool DEBUG_MAIN_ALG = false;

    /** Whether to show debug information for the svm */
    static const bool DEBUG_SVM= true;

	// OpenCV Documentation says that blocksize has to be 16x16 and cellsize 8x8. Other values are not supported.
	// Experiments say otherwise !?
	// blockssize and blockstride have to multiples of cellsize
	// image size has to be multiple of blocksize
    static const int HOG_CELLSIZE = 8;
    static const int HOG_BLOCKSTRIDE = 8;
    static const int HOG_BLOCKSIZE = 16;
    static const int HOG_BINS = 9;

	// Image will be resized to this size !
	// If the original size is not divideable by cellsize e.g.
    static const int HOG_IMAGE_SIZE_X = 96;
    static const int HOG_IMAGE_SIZE_Y = 96;

    static const int HOG_VIZ_SCALEFACTOR = 4;
    static constexpr double HOG_VIZ_VIZFACTOR = 3.0;

    /** Reduce features, 1 >= factor > 0 */
    static constexpr double PCA_REDUCTION_FACTOR = 1.0;

    static constexpr double SVM_C_VALUE = 0.1;

    /** Part of validation data */
    static const int DATESET_DIVIDER = 10;

};
