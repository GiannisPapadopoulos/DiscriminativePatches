#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace mai{
class facedetection
{
public:
	facedetection(void);
	~facedetection(void);
	static void faceDetec(vector<Mat*> &vImages);
};
}