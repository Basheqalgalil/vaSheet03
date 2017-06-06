#!/bin/bash

#block(name=train, threads=1, memory=10000, gpus=1, hours=24)

    mkdir -p log results

    ./executables/neural-network-trainer --config=config/training.config $OPTIONS


#block(name=forward, threads=1, memory=10000, gpus=1, hours=24)

    mkdir -p log results
    
    OPTIONS="--neural-network.load-model-from=results"

    ./executables/neural-network-trainer --config=config/forward.config $OPTIONS

