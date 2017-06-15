   
#!/bin/bash

#block(name=train, threads=1, memory=10000, gpus=1, hours=24)
   # prepare temporal 
    ../Nn/neural-network-trainer --config=config/PTtraining.config 

#training 
#block(name=train, threads=1, memory=10000, gpus=1, hours=24)

    mkdir -p log results

    ./executables/neural-network-trainer --config=config/task2Training.config 

