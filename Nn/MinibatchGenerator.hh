/*
 * MinibatchGenerator.hh
 *
 *  Created on: May 13, 2016
 *      Author: richard
 */

#ifndef NN_MINIBATCHGENERATOR_HH_
#define NN_MINIBATCHGENERATOR_HH_

#include <Core/CommonHeaders.hh>
#include <Nn/Types.hh>
#include <Nn/MatrixContainer.hh>
#include <Features/FeatureReader.hh>
#include <Features/AlignedFeatureReader.hh>
#include <Nn/FeatureTransformation.hh>

namespace Nn {

class MinibatchGenerator
{
private:
	static bool compare(const std::pair<u32, u32>& a, const std::pair<u32, u32>& b) { return a.first > b.first; }

	static const Core::ParameterEnum paramSourceType_;
	static const Core::ParameterEnum paramTargetType_;

private:
	FeatureType sourceType_;
	FeatureType targetType_;
	TrainingMode trainingMode_;

	Features::BaseFeatureReader* featureReader_;

	Matrix sourceBatch_;
	Matrix targetBatch_;
	MatrixContainer sourceSequenceBatch_;
	MatrixContainer targetSequenceBatch_;
	std::vector<u32> order_; // keep track of original sequence order
	u32 sourceDimension_;
	u32 targetDimension_;
	FeatureTransformation featureTransformation_;
	bool isInitialized_;
	bool generatedBatch_;

	void read(std::vector< Math::Matrix<Float> >& source, std::vector< Math::Matrix<Float> >& target);
	void generateSingleBatch(const std::vector< Math::Matrix<Float> >& source, const std::vector< Math::Matrix<Float> >& target);
	void generateSequenceBatch(const std::vector< Math::Matrix<Float> >& source, const std::vector< Math::Matrix<Float> >& target);

public:
	MinibatchGenerator(TrainingMode trainingMode);
	~MinibatchGenerator();

	void initialize();
	void generateBatch(u32 batchSize);
	/*
	 * @return the total number of feature vectors in the feature cache
	 */
	u32 totalNumberOfFeatures() const;
	/*
	 * @return the total number of sequences in the feature cache, else total number of features (same as above)
	 */
	u32 totalNumberOfObservations() const;
	FeatureType sourceType() const { return featureTransformation_.outputFormat(); }
	FeatureType targetType() const { return targetType_; }

	Matrix& sourceBatch();
	Matrix& targetBatch();

	MatrixContainer& sourceSequenceBatch();
	MatrixContainer& targetSequenceBatch();
	const std::vector<u32>& sequenceOrder() const { return order_; }
};

} // namespace

#endif /* NN_MINIBATCHGENERATOR_HH_ */
