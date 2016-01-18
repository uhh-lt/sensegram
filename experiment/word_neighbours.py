"""
Creates a DT file with closest neighbours of every word in the model
"""
#TODO: argument input as in postprocess.py of chinese-whispers
#IDEA: what if every word was embedded separately for its every possible pos tag? Like run#VB, run#N?


import codecs, pbar
from gensim.models import word2vec

model_path = 'model/text8_vectors.bin'
model_type = 'word2vec' # trained with word2vec or gensim
model_binary = True # True for binary, False for text (rare), only for word2vec
n_neighbours = 200
dt_output_path = 'dt/neighbours.txt'

print("Loading model...")
if model_type == 'word2vec':
	model = word2vec.Word2Vec.load_word2vec_format(model_path, binary=model_binary)
if model_type == 'gensim':
	model = word2vec.Word2Vec.load(model_path)
print("Vocabulary size: %i" % len(model.vocab))

print("Collecting word neighbours...")	
with codecs.open(dt_output_path, 'w', encoding='utf-8') as output:
	vocab_size = len(model.vocab)
	step = pbar.start_progressbar(vocab_size, 100)
	i = 0
	for word in model.index2word:
		neighbours = model.most_similar(positive=[word],topn=n_neighbours)
		for neighbour, sim in neighbours:
			output.write("%s\t%s\t%s\n" % (word, neighbour, str(sim)))
		if i%step==0:
			pbar.update_progressbar(i, vocab_size)
		i+=1
	pbar.finish_progressbar()	
	print("Number of processed words: %i" % i)		

# to apply clustering:
# cd ../chinese-whispers/			
# time java -Xms4G -Xmx4G -cp target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI -in ../experiment/dt/neighbours.txt -n 200 -N 200 -out ../experiment/dt/clusters.txt -clustering cw -e 0.01
#  -n <integer>           max. number of edges to process for each similar
#                         word (word subgraph connectivity)
#  -N <integer>           max. number of similar words to process for a
#                         given word (size of word subgraph to be clustered)

# to postprocess clusters:
# cd ../experiment
# time dt/postprocess.py -min_size 5 dt/clusters.txt