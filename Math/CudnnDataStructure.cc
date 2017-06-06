 /* CudnnDataStructure.cc
 *
 *  Created on: Sep 1, 2016
 *      Author: ahsan
 */
#include <Core/CommonHeaders.hh>
#include "CudnnDataStructure.hh"
#include "CudaDataStructure.hh"

using namespace Math;
using namespace Math::cuDNN;

bool CudnnDataStructure::isInitialized_ = false;
cudnnHandle_t CudnnDataStructure::cudnnHandle_;
#ifdef MODULE_CUDNN
cudnnTensorFormat_t CudnnDataStructure::tensorFormat_ = CUDNN_TENSOR_NCHW;
#endif

void CudnnDataStructure::initialize() {
	if(CudaDataStructure::hasGpu() && !isInitialized_) {
		cudnnStatus_t cudnnStatus;
		cudnnStatus = cuDNN::cuDNNCreateHandle(cudnnHandle_);
		if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
			std::cerr<<"Failed to create cuDNN handle"<<std::endl;
			exit(1);
		}
		isInitialized_ = true;
	}
}


/*
*/
