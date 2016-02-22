#!/bin/bash

SUB_DIR=2
START=50
END=100

#BSUB -J "KNN Computation"
#BSUB -eo ${output_dir}/stderr.txt
#BSUB -oo ${output_dir}/stdout.txt
#BSUB -n 14 
#BSUB -M 30000
#BSUB -x
#BSUB -W 3:00
#BSUB -q kurs3
#BSUB -u jonas.molinaramirez@stud.tu-darmstadt.de
#BSUB -N
cd ${output_dir}
python3 ${script} ${start} ${end} ${sub_dir} ${no_of_files}
