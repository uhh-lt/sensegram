#!/usr/bin/env python

import argparse
from os import listdir
from os.path import isfile, join, walk, split
from math import sqrt
import re
from pandas import read_csv


d = False
sense_file = "sense_inventory"
prediction_file = "predictions"

SEP = "@@"
VALSEP = ':'

ASSIGNED_SENSES_FILE = "./data/Senses-TWSI-2.csv"
INVENTORY_FILE = "./data/Inventory-TWSI-2.csv"

twsi_subst = dict()
sense_mappings = dict()
gold_labels = dict()
predictions = dict()
assigned_senses = dict()

"""
TwsiSubst class to store inventories
"""
class TwsiSubst:
    def __init__(self,word):
    	# target word
        self.word = word
        # related terms (with count) by sense id
        self.terms = {}
        # substitution counts by sense id
        self.scores = dict()
    
    def __eq__(self,other):
        return self.word == other
        
    def __hash__(self):
        return hash(self.word)
    
    # get substitutions for sense id
    def getSubst(self, num):
        return self.terms[num]
    
    # get scores by sense id
    def getScores(self, num):
        return self.scores[num]
    
    # add a new substiution with score
    def addTerm(self, num, term, count):
    	if not self.terms.has_key(num):
    	    self.terms[num] = dict()
    	    self.scores[num] = 0
    	if self.terms[num].has_key(term):
    	    self.terms[num][term] = int(self.terms[num][term]) + int(count)
    	else:
    	    self.terms[num][term] = int(count)
        self.scores[num] += int(count)
        
    # add a list of substitutions with score from string
    def addTerms(self, num, subs):
       subs_l = subs.split(',')
       for s in subs_l:
           s = s.strip()
           key, val = s.split(VALSEP)
           self.addTerm(num, key, val)    
    
    # list all sense ids
    def getSenseIds(self):
        return self.terms.keys()
    
    # does this sense id exist?
    def hasSenseId(self, num):
        return (num in self.scores)




""" 	loads all the senses which were assigned in TWSI 2.0
	assigned senses are stored in provided file
	list of senses is used to remove all other senses, since they are impossible substitutions for the TWSI task 
"""
def load_assigned_senses():
    print "Loading assigned TWSI senses..."
    global assigned_senses
    assigned_senses = set(line.strip() for line in open(ASSIGNED_SENSES_FILE))
    print "Loading done\n"



"""	loads all TWSI 2.0 senses from the TWSI dataset folder
	filters senses by removing senses which do not occur in the TWSI data
"""
def load_twsi_senses():
    print "Loading TWSI sense inventory..."
    substitutions = read_csv(INVENTORY_FILE, '/\t+/', encoding='utf8', header=None, engine="python") 
    #/\t+/   
    for i,s in substitutions.iterrows():
    	# create new TwsiSubst for the given word
    	#print s
    	word, t_id, subs = s[0].split('\t')
    	#print word, t_id, subs
    	#word, t_id, subs = s[0], s[1], s[2]
    	t_s = twsi_subst.get(word)
    	if t_s == None:
    	    t_s = TwsiSubst(word)
    	twsi_sense = word+SEP+t_id
        if twsi_sense not in assigned_senses:
            if d:
	        print "\nomitting TWSI sense "+twsi_sense+" as it did not occur in the sentences"
	    continue
        t_s.addTerms(t_id,subs)
        twsi_subst[word] = t_s
    print "\nLoading done\n"
    


""" 	loads custom sense inventory
	performs alignment using cosine similarity
"""
def load_sense_inventory(filename):
    print "Loading provided Sense Inventory "+filename+"..."
    mapping_f = "data/Mapping_"+split(INVENTORY_FILE)[1]+"_"+split(filename)[1]
    print "Mapping saved to "+mapping_f
    m_f = open(mapping_f, 'w')
    inventory = read_csv(filename, '/\t+/', encoding='utf8', header=None, engine="python")
    for r,inv in inventory.iterrows():
        word, ident, terms = inv[0].split('\t')
        if word in twsi_subst:
            m_f.write("\n\nProvided Sense:\t"+word+" "+ident+"\n")
            m_f.write("Inventory:\t"+terms+"\n")
            if d:
                print "\nSENSE: "+word+" "+ident
            twsi = twsi_subst.get(word)
            word_vec = dict()
            # split sense cluster into elements
            for el in set(terms.split(',')):
            	el = el.strip()
            	# split element into word and score
            	el_split = el.rsplit(VALSEP, 1)
            	# bug? changed from 'word' to 'word2'
            	word2 = el_split[0]
            	if len(el_split) > 1 and not re.match('\D+', el_split[1]):
            	    if word2 in word_vec:
            	        word_vec[word2] += float(el_split[1])
            	    else:
            	        word_vec[word2] = float(el_split[1])
            	else:
            	    word_vec[word2] = 1.0
            	    
            # matching terms to TWSI sense ids
            scores = dict()
            for i in twsi.getSenseIds():
            	twsi_sense = twsi.getSubst(i)
                scores[i] = calculate_cosine(twsi_sense, word_vec)
                m_f.write("\nTSWI Sense "+i+":\t")
                for key in twsi_sense.keys():
                    m_f.write(key+":"+str(twsi_sense[key])+", ")
                m_f.write("\nCosine Score:\t"+str(scores[i])+"\n")
                if d:
                    print "Score for ",i,":", scores[i]
               
            # assignment
            assigned_id = get_max_score(scores)
            sense_mappings[word+ident] = assigned_id
            if d:
                print "SCORES: "+str(scores)
                print "ASSIGNED ID: "+word+" "+ident+"\t"+str(assigned_id)
    print "\nLoading done\n"



""" 	loads and evaluates the results
"""
def evaluate_predicted_labels(filename):
    print "Evaluating Predicted Labels "+filename+"..."
    correct = 0
    retrieved = 0
    itemcount = 0
    checked = set()
    predictions = read_csv(filename, '/\t+/', encoding='utf8', header=0, engine="python")
    for i,p in predictions.iterrows():
         #print p[0]
         pred = p[0].split('\t')
         key = str(pred[0])+str(pred[1])
         gold = pred[4]
         oracle = pred[5]
         itemcount += 1
         if oracle == "":
             print "Sentence "+pred[0]+": Key '"+oracle+"' without sense assignment\n"
             #oracle_p = -1 bug?
             oracle = -1
         if key not in checked:
             checked.add(key)
             #print gold, oracle, sense_mappings[pred[1] + str(oracle)]
             #print sense_mappings
             if pred[1] + str(oracle) in sense_mappings and gold == sense_mappings[pred[1] + str(oracle)]:
                 correct += 1
             if int(oracle) > -1: 
                 retrieved += 1
             if d:  
             	 if oracle in sense_mappings:
             	     print "Sentence: "+key+"\tPrediction: "+oracle+"\tGold: "+key+"\tPredicted_TWSI_sense: "+str(sense_mappings[oracle])+"\tMatch:"+str(gold == sense_mappings[oracle])
             	 else:
             	     print "Sentence: "+key+"\tPrediction: "+oracle+"\tGold: "+key+"\tPredicted_TWSI_sense: "+"none"+"\tMatch: False"
                	     
         elif d:
             print "Sentence not in gold data: "+key+" ... Skipping sentence for evaluation."
    print "\nEvaluation done\n"
    return correct, retrieved, itemcount



""" 	gets maximum score from a dictionary
"""
def get_max_score(scores):
    max_value = 0
    max_id = -1
    for i in scores.keys():
        if scores[i] > max_value:
            max_value = scores[i]
            max_id = i
    return max_id



""" 	computes precision, recall and fscore
"""
def calculate_evaluation_scores(correct, retrieved, itemcount, eval_retrieved = False):
    if eval_retrieved:
        itemcount = retrieved
    precision = 0
    recall = 0
    if retrieved == 0:
        print "No predictions were retrieved!"
    else:
        precision = float(correct) / retrieved
    
    if itemcount == 0:
        print "No Gold labels, check TWSI path!"
    else:
        recall = float(correct) / itemcount
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore
    
        

""" 	computes cosine similarity between two vectors
"""
def calculate_cosine(v1, v2):
    score = 0
    len1 = 0
    len2 = 0
    for w in v1.keys():
        if w in v2.keys():
            if d:
                print "Element:",w, v1[w], v2[w]
            score += v1[w] * v2[w]
        len1 += pow(v1[w],2)
    for w in v2.keys():
        len2 += pow(v2[w],2)
    l1 = sqrt(len1)
    l2 = sqrt(len2)
    if l1 > 0 and l2 > 0:
        return score / (l1 * l2)
    return 0
    
    
""" 	6mcomputes the purity of a clustering
"""
def calculate_purity_clustering(c1, c2):
    # for clustering c1:
    	# get cluster
    	# calcucate purity(v1, v2)
    return 0
    	
def calculate_purity(v1, v2):
    #
    return 0
    



def main():
    global SEP, TWSI_PATH, d
    parser = argparse.ArgumentParser(description='Evaluation script for contextualizations with a custom Word Sense Inventory.')	
    parser.add_argument('sense_file', metavar='inventory', help='word sense inventory file, format:\n word_senseID <tab> list,of,words')
    parser.add_argument('predictions', help='word sense disambiguation predictions, format:\n sentenceID <tab> predicted-word_senseID')
    settings = parser.add_argument_group('Settings')
    settings.add_argument('-d', dest='debug', help='display debug output (default: False)', required=False)
    args = parser.parse_args()
    
    if args.debug:
        d = args.debug   
    
    load_assigned_senses()
    load_twsi_senses()
    load_sense_inventory(args.sense_file)
    correct, retrieved, count = evaluate_predicted_labels(args.predictions)
    
    print "\nEvaluation Results:"
    print "Correct, retrieved, nr_sentences"
    print correct, "\t", retrieved, "\t", count
    precision, recall, fscore = calculate_evaluation_scores(correct, retrieved, count)
    print "Precision:",precision, "\tRecall:", recall, "\tF1:", fscore
    print "Coverage: ", float(retrieved)/count
    

if __name__ == '__main__':
    main()

