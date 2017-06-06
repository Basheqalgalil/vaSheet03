/*
 * Utils.cpp
 *
 *  Created on: May 10, 2017
 *      Author: ahsan
 */
#include "Utils.hh"

void Utils::readVideo(const std::string& filename, std::vector<cv::Mat>& result) {
	// open video file
	cv::VideoCapture capture(filename);
	if(!capture.isOpened())
		Core::Error::msg() << "Unable to open Video: " << filename << Core::Error::abort;
	cv::Mat frame, tmp;
	result.clear();
	// read all frames
	u32 nFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
	while ((nFrames > 0) && (capture.read(frame))) {
		if (frame.channels() == 3)
			cv::cvtColor(frame, tmp, CV_BGR2GRAY);
		else
			tmp = frame;
		result.push_back(cv::Mat());
		tmp.convertTo(result.back(), CV_32FC1);

		//result.push_back(tmp);
		nFrames--;
	}
	capture.release();
}

void Utils::matrixToCVMat(const Math::Matrix<f32>& matrix, cv::Mat& cvMat, bool transpose) {

	u32 I = transpose ? matrix.nColumns() : matrix.nRows();
	u32 J = transpose ? matrix.nRows() : matrix.nColumns();

	for (u32 i=0; i<I; i++) {
		f32* row = cvMat.ptr<f32>(i);
		for (u32 j=0; j<J; j++) {
			if (transpose)
				row[j] = matrix.at(j, i);
			else
				row[j] = matrix.at(i, j);
		}
	}
}

void Utils::cvMatToMatrix(const cv::Mat& cvMat, Math::Matrix<f32>& matrix) {
	for (u32 i=0; i<(u32)cvMat.rows; i++) {
		const f32* row = cvMat.ptr<f32>(i);
		for (u32 j=0; j<(u32)cvMat.cols; j++) {
			matrix.at(i, j) = row[j];
		}
	}
}



