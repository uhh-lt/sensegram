#!/bin/bash


#BSUB -J "KNN Computation"
#BSUB -eo /home/kurse/jm18magi/sensegram/src/cython_test/cython2/stderr.txt
#BSUB -oo /home/kurse/jm18magi/sensegram/src/cython_test/cython2/stdout.txt
#BSUB -n 10 
#BSUB -M 10000
#BSUB -x
#BSUB -W 3:00
#BSUB -q kurs3
#BSUB -u jonas.molinaramirez@stud.tu-darmstadt.de
#BSUB -N
#BSUB -a openmp
module load gcc intel python/3
python3 load_job.py 
