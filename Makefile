export LANG:=en_US.UTF-8
export SHELL:=/bin/bash


install:
	pip install -r requirements.txt
	python -m spacy download en
	make install-faiss
	bash 

install-ubuntu-16-04:
	sudo apt-get update
	sudo apt-get install swig libopenblas-dev python-dev gcc g++ python3-pip unzip
	make install-anaconda3
	make install 

download:
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.w2v
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.w2v.probs
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.words
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.contexts
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.jbt
	wget -P model http://panchenko.me/data/joint/sensegram/wiki.senses.jbt.probs
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.words
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.contexts
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.w2v
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.w2v.probs
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.jbt
	wget -P model http://panchenko.me/data/joint/sensegram/ukwac.senses.jbt.probs

train:
	bash train.sh

train-wikipedia-sample:
	wget http://panchenko.me/data/joint/corpora/wiki.txt.gz -P model
	bash train.sh model/wiki.txt.gz

train-wikipedia:
	wget http://panchenko.me/data/joint/corpora/en59g/wikipedia.txt.gz -P model
	bash train.sh model/wikipedia.txt

install-faiss:
	rm -rf faiss
	git clone https://github.com/facebookresearch/faiss.git 
	# for compilation using other linux distributions see the faiss/makefile.inc and change it accordingly
	cp makefile-faiss-ubuntu-16-04-python3.inc faiss/makefile.inc
	make -C faiss
	make -C faiss py
	mv faiss faiss-src
	mv -fv faiss-src/{faiss.py,swigfaiss.py,_swigfaiss.so} .

clean:
	rm -rf faiss
	rm -rf model

install-anaconda3:
	wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
	bash ~/anaconda.sh -b -p ${HOME}/anaconda
	echo 'export PATH="${HOME}/anaconda/bin:${PATH}"' >> ~/.bashrc	
	echo 'source ${HOME}/anaconda/bin/activate' >> ~/.bashrc
