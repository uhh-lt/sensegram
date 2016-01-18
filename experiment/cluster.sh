#!/bin/bash
# implementation of chinese-whispers (the folder) should be next to experiment forlder (or indicate the path to the jar yourself) 
time java -Xms4G -Xmx4G -cp ../chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI -in dt/neighbours.txt -n 200 -N 200 -out dt/clusters.txt -clustering cw -e 0.01
