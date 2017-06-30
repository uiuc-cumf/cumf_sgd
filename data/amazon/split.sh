#!/bin/bash

shuf -n 22507155 ratings_Books > ratings_Books_shuf
head -n 225071 ratings_Books_shuf > ratings_Books_test
tail -n +225072 ratings_Books_shuf > ratings_Books_train

shuf -n 7824482 ratings_Electronics > ratings_Electronics_shuf
head -n 78245 ratings_Electronics_shuf > ratings_Electronics_test
tail -n +78246 ratings_Electronics_shuf > ratings_Electronics_train

shuf -n 4607047 ratings_Movies_and_TV > ratings_Movies_and_TV_shuf
head -n 46070 ratings_Movies_and_TV_shuf > ratings_Movies_and_TV_test
tail -n +46071 ratings_Movies_and_TV_shuf > ratings_Movies_and_TV_train