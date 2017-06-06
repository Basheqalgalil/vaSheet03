/*
 * Gmm.h
 *
 *  Created on: May 12, 2017
 *      Author: ahsan
 */

#ifndef ACTIONRECOGNITION_GMM_HH_
#define ACTIONRECOGNITION_GMM_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include <opencv2/ml.hpp>
#include "Utils.hh"

namespace ActionRecognition {

class Gmm
{
private:
	static const Core::ParameterInt paramNumberOfGaussians_;
	static const Core::ParameterInt paramNumberOfIterations_;
	static const Core::ParameterString paramModelFile_;

	u32 nGaussians_;
	u32 nIterations_;
	std::string modelFile_;

	cv::EM gmm_;


	void initialize(const Math::Matrix<Float>& trainingData);

public:
	Gmm();
	virtual ~Gmm() {}

	/*
	 * @param training data: a matrix of size <feature-dimension> x <number-of-training-samples> (rows: dimension, cols: samples)
	 */
	void train(const Math::Matrix<Float>& trainingData);

	/*
	 * @param data: a vector of size <feature-dimension> x 1 or 1 x <feature-dimension>
	 * @param result: a vector of size 1 x <feature-dimension>
	 */
	void predict(const Math::Vector<f32>& vec, Math::Vector<f32>& result);

	void save();
	void load();

	void getMeans(Math::Matrix<f32>& result);
	void getSigmas(Math::Matrix<f32>& result);
	void getWeights(Math::Vector<f32>& result);

};

} // namespace



#endif /* ACTIONRECOGNITION_GMM_HH_ */
