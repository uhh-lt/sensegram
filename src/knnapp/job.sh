#!/bin/bash

SUB_DIR=2
START=50
END=100

#BSUB -J "KNN Computation"
#BSUB -eo /home/kurse/jm18magi/sensegram/src/knnapp/output/stderr.txt
#BSUB -oo /home/kurse/jm18magi/sensegram/src/knnapp/output/stdout.txt
#BSUB -n 14 
#BSUB -M 30000
#BSUB -x
#BSUB -W 3:00
#BSUB -q kurs3
#BSUB -u jonas.molinaramirez@stud.tu-darmstadt.de
#BSUB -N

python3 memtest.py 
