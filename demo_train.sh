#!/usr/bin/env bash

if [ ! -e model ]; then
	mkdir model 
fi 

if [ ! -e model/text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -P model
  unzip model/text8.zip -d model
fi

python train.py model/text8 -cbow 1 -size 100 -window 3 -threads 4 -iter 3 -min_count 5 -only_letters -n 200 -N 200 -min_size 5
