#!/bin/bash

#BSUB -J "predict TWSI w1cb1 upper"
#BSUB -eo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/experiment_17.02_6models_jobs/job_w1cb1_TWSI_prediction_upper_err.txt
#BSUB -oo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/experiment_17.02_6models_jobs/job_w1cb1_TWSI_prediction_upper_out.txt
#BSUB -n 14
#BSUB -M 10000
#BSUB -W 700
#BSUB -x
#BSUB -q kurs3

cd /work/scratch/kurse/zu66ytyq/thesis/experiment/
time python3 prediction.py context-eval/data/Dataset-TWSI-2.csv model/res1/corpus_en.norm-sz100-w1-cb1-it1-min20.w2v.uppercase.senses model/res1/corpus_en.norm-sz100-w1-cb1-it1-min20.w2v.contexts eval/res1/corpus_en.norm-sz100-w1-cb1-it1-min20_TWSI-2_predictions_nothr_upper.csv -lowercase

