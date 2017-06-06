/*
 * Gmm.cc
 *
 *  Created on: May 12, 2017
 *      Author: ahsan
 */

#include "Gmm.hh"
#include <math.h>
#include "Math/Random.hh"
using namespace ActionRecognition;


const Core::ParameterInt Gmm::paramNumberOfGaussians_("number-of-gaussians", 64, "gmm");
const Core::ParameterInt Gmm::paramNumberOfIterations_("number-of-iterations", 100, "gmm");
const Core::ParameterString Gmm::paramModelFile_("model-file", "", "gmm");

Gmm::Gmm() :
		nGaussians_(Core::Configuration::config(paramNumberOfGaussians_)),
		nIterations_(Core::Configuration::config(paramNumberOfIterations_)),
		modelFile_(Core::Configuration::config(paramModelFile_)),
		gmm_(nGaussians_, cv::EM::COV_MAT_DIAGONAL) {

}

void Gmm::initialize(const Math::Matrix<Float>& trainingData) {
	//gmm_.set("covMatType", cv::EM::COV_MAT_DIAGONAL);
	//gmm_.set("nclusters", (int)nGaussians_);
	//gmm_.set("maxIters", (int)nIterations_);
}


void Gmm::train(const Math::Matrix<Float>& trainingData) {
	cv::Mat cvMatData(trainingData.nColumns(), trainingData.nRows(), CV_32FC1);
	Utils::matrixToCVMat(trainingData, cvMatData, true);

	gmm_.train(cvMatData);


	save();
}

void Gmm::predict(const Math::Vector<f32>& vec, Math::Vector<f32>& result/*const cv::Mat& vec, cv::Mat& result*/) {
	cv::Mat cvVec(1, vec.nRows(), CV_32FC1);
	cv::Mat cvResult;
	for (u32 i=0; i<result.nRows(); i++) {
		cvVec.at<f32>( 0, i) = result.at(i);
	}

	gmm_.predict(cvVec, cvResult);
	result.resize(cvResult.cols);
	for (u32 j=0; j<cvResult.cols; j++) {
		result.at(j) = (f32)cvResult.at<f64>(0, j);
	}
}

void Gmm::save() {
	if(modelFile_.empty()) {
		Core::Error::msg("gmm.model-file must not be empty.") << Core::Error::abort;
	}

	cv::FileStorage fs(modelFile_, cv::FileStorage::WRITE);
	if (fs.isOpened()) {
		gmm_.write(fs);
		fs.release();
	}
}

void Gmm::load() {
	if(modelFile_.empty()) {
			Core::Error::msg("gmm.model-file must not be empty.") << Core::Error::abort;
	}

	cv::FileStorage fs(modelFile_, cv::FileStorage::READ);
	if (fs.isOpened()) {
		const cv::FileNode& fn = fs["StatModel.EM"];
		gmm_.read(fn);
		fs.release();
	}

	Math::Matrix<f32> means;
	cv::Mat emMean = gmm_.getMat("means");
	means.resize(emMean.rows, emMean.cols);
	Utils::cvMatToMatrix(emMean, means);
	means.write("/home/ahsan/temp/gmm/means.txt");
}

void Gmm::getMeans(Math::Matrix<f32>& result) {
	cv::Mat emMean = gmm_.getMat("means");
	result.resize(emMean.cols, emMean.rows);

	for (u32 i=0; i<result.nColumns(); i++) {
		f64 *meanRow = emMean.ptr<f64>(i);
		for (u32 j=0; j<result.nRows(); j++) {
			result.at(j,i) = (f32)meanRow[j];
		}
	}
}
void Gmm::getSigmas(Math::Matrix<f32>& result) {
	std::vector<cv::Mat> sigmas = gmm_.getMatVector("covs");
	result.resize(sigmas.at(0).rows, sigmas.size());

	for (u32 i=0; i<sigmas.size(); i++) {
		for (u32 j=0; j<sigmas.at(i).rows; j++) {
			result.at(j, i) = (f32)sigmas.at(i).at<f64>(j, j);
		}
	}
}
void Gmm::getWeights(Math::Vector<f32>& result) {
	cv::Mat weights = gmm_.getMat("weights");
	result.resize(weights.cols);
	for (u32 i=0; i<result.nRows(); i++) {
		result.at(i) = (f32)weights.at<f64>(0, i);
	}
}
