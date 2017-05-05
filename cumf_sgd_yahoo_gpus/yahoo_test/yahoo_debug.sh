#! /bin/bash

#echo "test"


#nvprof --print-gpu-trace   


#nvprof --metrics gld_throughput,gst_throughput 


#nvprof --metrics dram_read_throughput,dram_write_throughput 



#/home/cli99/cumf_sgd/cumf_sgd-moha/singleGPU/cumf_sgd -g 6 -l 1 -a 0.08 -b 0.20 -u 8 -v 8 -x 1 -y 1 -s 1792 -k 128 -t 18 /home/cli99/cumf_sgd/data/yahoo/yahoo.train.bin


#nvprof -f -o timeline_4_4_4 
../sgd_gpu -g 6 -l 1 -a 0.08 -b 0.20 -gpus 2 -u 4 -v 4 -x 1 -y 1 -s 1792 -k 128 -t 20 /home/cli99/cumf_sgd/data/yahoo/yahoo.train.bin
#nvprof -f -o analysis_4_4_4 --analysis-metrics   ../sgd_gpu -g 6 -l 1 -a 0.08 -b 0.20 -gpus 4 -u 4 -v 4 -x 1 -y 1 -s 1792 -k 128 -t 20 yahoo.train.bin




/home/cli99/cumf_sgd/test/mf-predict -e 0 /home/cli99/cumf_sgd/data/yahoo/yahoo.test.bin yahoo.train.bin.model

#rm -f yahoo.train.bin.model

