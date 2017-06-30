#! /bin/bash

../../test/mf-train -l2 0.05 -k 100 -t 30 ratings_Books_train.bin ratings_Books_train_libmf.bin.model
# ../../test/mf-predict -e 0 ratings_Books_test.bin ratings_Books_train_libmf.bin.model