/*
 * CudnnDataStructure.hh
 *
 *  Created on: Sep 1, 2016
 *      Author: ahsan
 */

#ifndef MATH_CUDNNDATASTRUCTURE_HH_
#define MATH_CUDNNDATASTRUCTURE_HH_

#include "CudnnWrapper.hh"
#include "CudaMatrix.hh"
namespace Math {

namespace cuDNN {

class CudnnDataStructure {
private:
	static bool isInitialized_;
public:
	static cudnnTensorFormat_t tensorFormat_;
	static cudnnHandle_t cudnnHandle_;

	static void initialize();
};

} //namespace cuDNN

}//namespace Math



#endif /* MATH_CUDNNDATASTRUCTURE_HH_ */
