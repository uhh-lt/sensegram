""" Fills out the TWSI dataset with predictions. Requires sense and context vectors.
If a word to disambiguate is not in the sense model -> no prediction
If a context is not in the context model -> don't use it in computation
All context words are lowercased and filtered from words with other chars than [a-z] before windowing
"""
from wsd import WSD
from pandas import read_csv
import pbar

print("Loading models...")
wsd_model = WSD("model/wiki_sense_vectors.bin", "model/wiki_vectors.bin.contexts")
TWSI_file_to_fill = "contextualization-eval/data/Dataset-TWSI-2.csv"
output = "contextualization-eval/data/predictions.csv"
n_neighbours = 50

print("Loading test set...")
#header 'context_id\ttarget\ttarget_pos\ttarget_position\tgold_sense_ids\tpredict_sense_ids\tgolden_related\tpredict_related\tcontext'
reader = read_csv(TWSI_file_to_fill, encoding="utf-8", delimiter="\t", dtype={'predict_related': object, 'gold_sense_ids':object, 'predict_sense_ids':object})
rows_count = reader.shape[0]
print(str(rows_count) + " test instances")
step = pbar.start_progressbar(rows_count, 100)

for i, row in reader.iterrows():
	sense_id = wsd_model.disambiguate(row.context, row.target_position, row.target)
	if sense_id is None:
		continue
	
	reader.set_value(i, 'predict_sense_ids', sense_id)
	#reader['predict_sense_ids'][i] = sense_id
	
	word_sense = row.target+'#'+str(sense_id)
	neighbours = wsd_model.vs.most_similar(word_sense, topn=n_neighbours)
	neigh = []
	for neighbour, sim in neighbours:
		n, n_sid = neighbour.split("#")
		neigh.append("%s:%.3f" % (n, float(sim)))
	
	reader.set_value(i, 'predict_related', ",".join(neigh))
	#reader['predict_related'][i] = ",".join(neigh)
	
	if i%step==0:
		pbar.update_progressbar(i, rows_count)
pbar.finish_progressbar()

reader.to_csv(sep='\t', path_or_buf=output, encoding="utf-8", index=False)
print("Saved predictions to " + output)

# to run evaluation:
# cd contextualization-eval/
# time python twsi_evaluation.py ../dt/inventory.csv data/predictions.csv 

