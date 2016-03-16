#!/bin/bash

#BSUB -J "predict TWSI w1cb0"
#BSUB -eo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/experiment_17.02_6models_jobs/job_w1cb0_TWSI_prediction_err.txt
#BSUB -oo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/experiment_17.02_6models_jobs/job_w1cb0_TWSI_prediction_out.txt
#BSUB -n 12
#BSUB -M 10000
#BSUB -W 700
#BSUB -x
#BSUB -q kurs3

cd /work/scratch/kurse/zu66ytyq/thesis/experiment/
time python3 prediction.py contextualization-eval/data/Dataset-TWSI-2.csv model/corpus_en.norm-sz100-w1-cb0-it1-min20.w2v.senses model/corpus_en.norm-sz100-w1-cb0-it1-min20.w2v.contexts eval/corpus_en.norm-sz100-w1-cb0-it1-min20_TWSI-2_predictions_nothr.csv -lowercase
