/*
 * Preprocessor.hh
 *
 *  Created on: May 23, 2017
 *      Author: richard
 */

#ifndef NN_PREPROCESSOR_HH_
#define NN_PREPROCESSOR_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "MatrixContainer.hh"

namespace Nn {

class FeatureTransformation
{
private:
	static const Core::ParameterEnum paramTransformationType_;
	static const Core::ParameterInt paramTransformedFeatureDimension_;
public:
	enum TransformationType { none, vectorToSequence, sequenceToVector };
protected:
	TransformationType type_;
	u32 dimension_;
	FeatureType originalType_;
public:
	FeatureTransformation(FeatureType originalType);
	virtual ~FeatureTransformation() {}

	void transform(Matrix& in, MatrixContainer& out);
	void transform(MatrixContainer& in, Matrix& out);

	FeatureType outputFormat() const;
};

} // namespace

#endif /* NN_PREPROCESSOR_HH_ */
