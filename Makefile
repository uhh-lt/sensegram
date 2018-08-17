export LANG:=en_US.UTF-8
export SHELL:=/bin/bash


install:
	pip install -r requirements.txt
	python -m spacy download en
	make install-faiss

install-with-anaconda:
	make install-anaconda3
	make install 

train:
	bash train.sh

train-wikipedia-sample:
	wget http://panchenko.me/data/joint/corpora/wiki.txt.gz -P model
	bash train.sh model/wiki.txt.gz

train-wikipedia:
	wget http://panchenko.me/data/joint/corpora/en59g/wikipedia.txt.gz -P model
	bash train.sh model/wikipedia.txt.gz

install-faiss:
	conda install faiss-cpu -c pytorch

install-faiss-sources:
	sudo apt-get update
	sudo apt-get install swig libopenblas-dev python-dev gcc g++ python3-pip unzip
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
	source ${HOME}/anaconda/bin/activate

extract-text:
	wget -O model/wiki.xml.bz2 https://dumps.wikimedia.org/dewiki/20180120/dewiki-20180120-pages-articles-multistream.xml.bz2 
	git clone https://github.com/attardi/wikiextractor.git
	python wikiextractor/WikiExtractor.py model/wiki.xml.bz2 -o model/wiki --discard_elements gallery,timeline,noinclude --processes $$(nproc) --filter_disambig_pages -b 100M
	cat model/wiki/AA/wiki* > model/wiki.txt	
	sed -i 's/<[^>]*>/ /g' model/wiki.txt
