/*
 * Types.hh
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

#ifndef NN_TYPES_HH_
#define NN_TYPES_HH_

#include <Core/Types.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Math/CudnnConvolution.hh>
#include <Math/CudnnPooling.hh>
#include <Math/CudnnBatchNormalization.hh>

namespace Nn {

// feature types, training task, and training modes
enum TrainingTask { classification, regression };
enum FeatureType { single, sequence };
enum TrainingMode { supervised, unsupervised };

// neural network matrix/vector types
typedef Math::CudaMatrix<Float> Matrix;
typedef Math::CudaVector<Float> Vector;
#ifdef MODULE_CUDNN
typedef Math::cuDNN::CudnnConvolution<Float> CudnnConvolution;
typedef Math::cuDNN::CudnnPooling<Float> CudnnPooling;
typedef Math::cuDNN::CudnnBatchNormalization<Float> CudnnBatchNormalization;
#endif

} // namespace

#endif /* NN_TYPES_HH_ */
