# This is for Centos:
# # BLASLDFLAGS?=/usr/lib64/libopenblas.so.0
#
# # for Ubuntu 16:
# # sudo apt-get install swig libopenblas-dev python-numpy python-dev
# # BLASLDFLAGS?=/usr/lib/libopenblas.so.0
#
# # for Ubuntu 14:
# # sudo apt-get install swig libopenblas-dev liblapack3 python-numpy python-dev
# # BLASLDFLAGS?=/usr/lib/libopenblas.so.0 /usr/lib/lapack/liblapack.so.3.0

install:
	sudo pip install -r requirements.txt
	sudo python -m spacy download en

download:
	wget http://panchenko.me/data/joint/sensegram/wiki.senses.w2v
	wget http://panchenko.me/data/joint/sensegram/wiki.senses.w2v.probs
	wget http://panchenko.me/data/joint/sensegram/wiki.words
	wget http://panchenko.me/data/joint/sensegram/wiki.contexts
	wget http://panchenko.me/data/joint/sensegram/wiki.senses.jbt
	wget http://panchenko.me/data/joint/sensegram/wiki.senses.jbt.probs
	wget http://panchenko.me/data/joint/sensegram/wiki.senses.twsi
	wget http://panchenko.me/data/joint/sensegram/wiki.senses.twsi.probs
	wget http://panchenko.me/data/joint/sensegram/ukwac.words
	wget http://panchenko.me/data/joint/sensegram/ukwac.contexts
	wget http://panchenko.me/data/joint/sensegram/ukwac.senses.w2v
	wget http://panchenko.me/data/joint/sensegram/ukwac.senses.w2v.probs
	wget http://panchenko.me/data/joint/sensegram/ukwac.senses.jbt
	wget http://panchenko.me/data/joint/sensegram/ukwac.senses.jbt.probs
	wget http://panchenko.me/data/joint/sensegram/ukwac.senses.twsi
	wget http://panchenko.me/data/joint/sensegram/ukwac.senses.twsi.probs

train:
	bash demo_train.sh

