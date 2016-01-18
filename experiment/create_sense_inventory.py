""" Create word sense inventory for contextualization evaluation
Format: Word<Tab>SenseID<Tab>list:5,of:3,related:1,words:1"""

#TODO: argument input as in postprocess.py of chinese-whispers
#IDEA: We actually collect sense neighbours (not neighbours). For the inventory, however, we have to strip the sense id.
# And these "normal" word embeddings now characterise our sense. Is it correct? Maybe search for neighbours in the original word model? 


import codecs, pbar
from gensim.models import word2vec

model_path = 'model/text8_sense_vectors.bin'
model_type = 'word2vec' # trained with word2vec or gensim
model_binary = True # True for binary, False for text (rare), only for word2vec
n_neighbours = 50
dt_output_path = 'dt/sanity_check.txt'
inventory_path = 'dt/inventory.csv'

print("Loading model...")
if model_type == 'word2vec':
	model = word2vec.Word2Vec.load_word2vec_format(model_path, binary=model_binary)
if model_type == 'gensim':
	model = word2vec.Word2Vec.load(model_path)
print("Vocabulary size: %i" % len(model.vocab))

print("Collecting word neighbours...")	
with codecs.open(dt_output_path, 'w', encoding='utf-8') as san_output:
	with codecs.open(inventory_path, 'w', encoding='utf-8') as inv_output:
		vocab_size = len(model.vocab)
		step = pbar.start_progressbar(vocab_size, 100)
		i = 0
		for word_sense, voc in sorted(model.vocab.items()):
			neighbours = model.most_similar(positive=[word_sense],topn=n_neighbours)
			neigh = []
			word, sense_id = word_sense.split("#")
			for neighbour, sim in neighbours:
				san_output.write("%s\t%s\t%s\n" % (word_sense, neighbour, str(sim)))
				n, n_sid = neighbour.split("#")
				neigh.append("%s:%.3f" % (n, float(sim))) 
			if i%step==0:
				pbar.update_progressbar(i, vocab_size)
			i+=1
			inv_output.write("%s\t%s\t%s\n" % (word, sense_id, ",".join(neigh)))
		pbar.finish_progressbar()	
		print("Number of processed words: %i" % i)		
