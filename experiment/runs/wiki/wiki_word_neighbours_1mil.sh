#!/bin/bash

#BSUB -J "collect neighbours all"
#BSUB -eo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/wiki/wiki_word_neighbours_1mil_err.txt
#BSUB -oo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/wiki/wiki_word_neighbours_1mil_out.txt
#BSUB -n 16
#BSUB -M 10000
#BSUB -W 700
#BSUB -x
#BSUB -q kurs3

cd /work/scratch/kurse/zu66ytyq/thesis/experiment/
time python3 word_neighbours.py model/wiki-sz300-w3-cb1-it3-min20.w2v intermediate/wiki_neighbours_1mil.csv -vocab_limit 1000000