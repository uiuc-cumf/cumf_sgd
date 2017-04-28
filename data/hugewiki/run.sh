#! /bin/bash

../../singleGPU/cumf_sgd -g 1 -l 0.03 -a 0.08 -b 0.3 -u 8 -v 1 -x 8 -y 1 -s 1792 -k 128 -t 20 ./hugewiki.bin.train
