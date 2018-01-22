#!/usr/bin/env bash

if [ ! -e model ]; then
	mkdir model 
fi 

if [ -z "$1" ]; then 
    if [ ! -e model/text8  ]; then
      wget http://mattmahoney.net/dc/text8.zip -P model
      unzip model/text8.zip -d model
    fi
    corpus=model/text8
else
    corpus=$1
fi

python train.py $corpus -cbow 1 -size 100 -window 5 -threads $(nproc) -iter 3 -min_count 10 -only_letters -n 200 -N 200 -min_size 5
