/*
 * Application.hh
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#ifndef NN_APPLICATION_HH_
#define NN_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"
#include "Types.hh"

namespace Nn {

class Application: public Core::Application
{
public:
	Application() {}
	virtual ~Application() {}
	void main();
};

class NeuralNetworkApplication {
private:
	static const Core::ParameterEnum paramAction_;
	static const Core::ParameterInt paramBatchSize_;
    enum Action { none, training, forwarding,prpareTemporal };
	u32 batchSize_;
private:
	void initialize();

public:
	NeuralNetworkApplication();
	virtual ~NeuralNetworkApplication() {}
	virtual void run();
};

} // namespace

#endif /* NN_APPLICATION_HH_ */
