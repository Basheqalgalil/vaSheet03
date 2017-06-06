/*
 * Utils.hh
 *
 *  Created on: May 10, 2017
 *      Author: ahsan
 */

#ifndef ACTIONRECOGNITION_UTILS_HH_
#define ACTIONRECOGNITION_UTILS_HH_
#include "Core/CommonHeaders.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml.hpp>
#include "Math/Matrix.hh"
#include "Math/Vector.hh"


class Utils {
public:
	static void readVideo(const std::string& filename, std::vector<cv::Mat>& result);
	static void cvMatToMatrix(const cv::Mat& cvMat, Math::Matrix<f32>& matrix);
	static void matrixToCVMat(const Math::Matrix<f32>& matrix, cv::Mat& cvMat, bool transpose = false);
};




#endif /* ACTIONRECOGNITION_UTILS_HH_ */
