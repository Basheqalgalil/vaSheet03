/*
 * Application.cc
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#include <iostream>
#include "Application.hh"
#include "Trainer.hh"
#include "Forwarder.hh"

using namespace Nn;

APPLICATION(Nn::Application)

const Core::ParameterEnum NeuralNetworkApplication::paramAction_("action", "none, training, forwarding", "none" , "");

const Core::ParameterInt NeuralNetworkApplication::paramBatchSize_("batch-size", 1, "");

void Application::main() {
	NeuralNetworkApplication app;
	app.run();
}

NeuralNetworkApplication::NeuralNetworkApplication() :
		batchSize_(Core::Configuration::config(paramBatchSize_))
{
	// batch size must be at least 1
	require_ge(batchSize_, 1);
}

void NeuralNetworkApplication::run() {
	switch ((Action)Core::Configuration::config(paramAction_)) {
	case training:
	{
		Trainer* trainer = Trainer::createTrainer();
		trainer->initialize();
		trainer->processAllEpochs(batchSize_);
		trainer->finalize();
	}
	break;
	case forwarding:
	{
		Forwarder forwarder;
		forwarder.initialize();
		forwarder.forward(batchSize_);
		forwarder.finalize();
	}
	break;
	case none:
	default:
		Core::Error::msg("No action given.") << Core::Error::abort;
		break;
	}
}
