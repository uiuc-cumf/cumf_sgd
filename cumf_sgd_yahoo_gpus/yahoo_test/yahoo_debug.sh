#! /bin/bash

#echo "test"


#nvprof --print-gpu-trace   


#nvprof --metrics gld_throughput,gst_throughput 


#nvprof --metrics dram_read_throughput,dram_write_throughput 



/home/cli99/cumf_sgd/singleGPU/cumf_sgd -g 6 -l 1 -a 0.08 -b 0.20 -u 1 -v 1 -x 1 -y 1 -s 1792 -k 128 -t 30 /home/cli99/cumf_sgd/data/yahoo/yahoo.train.bin

/home/cli99/cumf_sgd/test/mf-predict -e 0 /home/cli99/cumf_sgd/data/yahoo/yahoo.test.bin yahoo.train.bin.model

#rm -f yahoo.train.bin.model

