export LANG:=en_US.UTF-8
export SHELL:=/bin/bash


install:
	pip install -r requirements.txt
	python -m spacy download en
	make faiss

install-ubuntu-16-04:
	sudo apt-get update
	sudo apt-get install swig libopenblas-dev python-dev gcc g++ python3-pip unzip
	make install 

download:
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.w2v
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.w2v.probs
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.words
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.contexts
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.jbt
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.jbt.probs
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.twsi
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.twsi.probs
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.words
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.contexts
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.w2v
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.w2v.probs
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.jbt
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.jbt.probs
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.twsi
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.twsi.probs

train:
	bash demo_train.sh

faiss:
	rm -rf faiss
	git clone https://github.com/facebookresearch/faiss.git 
	# for compilation using other linux distributions see the faiss/makefile.inc and change it accordingly
	cp makefile-faiss-ubuntu-16-04-python3.inc faiss/makefile.inc
	make -C faiss all py

clean:
	rm -rf faiss
	rm -rf model
