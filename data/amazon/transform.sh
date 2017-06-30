#!/bin/bash

if [ ! -f transform ]; then
    g++ transform.cpp -o transform
fi

./transform ratings_Books_test
./transform ratings_Books_train

./transform ratings_Electronics_test
./transform ratings_Electronics_train

./transform ratings_Movies_and_TV_test
./transform ratings_Movies_and_TV_train
