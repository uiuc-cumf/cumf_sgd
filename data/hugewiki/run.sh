#! /bin/bash

#./../singleGPU/cumf_sgd -g 1 -l 0.03 -a 0.08 -b 0.3 -u 1 -v 8 -x 1 -y 8 -s 1792 -k 128 -t 20 ./hugewiki.bin.train

RES=res_e.out

if [ -f $RES ]; then
    rm $RES
fi

for (( c=1; c<=40; c+=1 ))
do
        echo $c
        echo -n -e "$c," >> $RES
        ../../singleGPU/cumf_sgd -g 1 -l 0.03 -a 0.08 -b 0.3 -u 1 -v 8 -x 1 -y 8 -s 1792 -k 128 -t $c ./hugewiki.bin.train > run.out
        ../../test/mf-predict -e 0 hugewiki.bin.test hugewiki.bin.train.model > test.out
        echo "finish run"

        sed -n 's/^RMSE[[:space:]]*= //p' test.out | tr '\n' ',' >> $RES
        sed -n 's/^sgd_update_k128[[:space:]]*://p' run.out | tr '\n' ',' >> $RES
        sed -n 's/^update_per_sec[[:space:]]*://p' run.out >> $RES
done

