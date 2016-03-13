#!/bin/bash

#BSUB -J "predict TWSI w8cb1"
#BSUB -eo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/experiment_17.02_6models_jobs/job_1.5wiki_ukwac_TWSI_prediction_err.txt
#BSUB -oo /work/scratch/kurse/zu66ytyq/thesis/experiment/runs/experiment_17.02_6models_jobs/job_1.5wiki_ukwac_TWSI_prediction_out.txt
#BSUB -n 16
#BSUB -M 8000
#BSUB -W 400
#BSUB -x
#BSUB -q kurs3

cd /work/scratch/kurse/zu66ytyq/thesis/experiment/
time python3 ./prediction.py context-eval/data/Dataset-TWSI-2.csv model/1.5wiki_sense_vectors_ukwac.bin model/1.5wiki_context_vectors.bin eval/1.5wiki_ukwac_TWSI-2_predictions_nothr.csv -lowercase