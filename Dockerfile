FROM rai/nccl:8.0
MAINTAINER Cheng <cli99@illinois.edu>

WORKDIR /usr/local
RUN git clone https://github.com/cuMF/cumf_sgd

WORKDIR /usr/local/cumf_sgd/data/netflix/
RUN curl -fsSL http://www.select.cs.cmu.edu/code/graphlab/datasets/netflix_mme -O

RUN cd /usr/local/cumf_sgd/data/netflix && \
    make transform && \
    ./transform netflix_mme

WORKDIR /data
RUN mv /usr/local/cumf_sgd/data/netflix/netflix_mme.bin /data && \
    rm -fr /usr/local/cumf_sgd

