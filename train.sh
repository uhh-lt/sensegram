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

python train.py $corpus -size 100 -iter 3 
