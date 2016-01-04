//#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace mai{
class facedetection
{
public:
	facedetection(void);
	virtual~facedetection(void);
	static void faceDetec(std::vector<cv::Mat*> &vImages);
};
}