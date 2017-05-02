#! /bin/bash

../../singleGPU/cumf_sgd -g 1 -l 1.0 -a 0.08 -b 0.2 -u 1 -v 1 -x 1 -y 1 -s 1792 -k 128 -t 5 ./yahoo.train.bin

#../../singleGPU/cumf_sgd_rpcs -g 1 -l 1.0 -a 0.08 -b 0.2 -u 1 -v 1 -x 1 -y 1 -s 1792 -k 128 -t 5 ./yahoo.train.bin
