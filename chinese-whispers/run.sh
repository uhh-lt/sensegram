#!/bin/bash

if [ -z "$1" ] ; then
    echo "Run noun sense induction"
    echo "usage: ./run.sh <dt-csv-file>"
    echo "dt file should be in the format: word1<TAB>word2<TAB>sim"
    exit
fi

./build.sh

e=0  # min weight of edge
input=$1 

JAVA_CLASSPATH=$JAVA_CLASSPATH:`cat .dependency-jars` &&
JAVA_CLASSPATH=$JAVA_CLASSPATH:`echo target/*.jar | tr " " ":"` &&

for method in cw mcl ; do
    for N in 200 50 100 500; do # 20 50 100 200 500; do
        for n in 200 50 100 500 ; do # 5 10 20 50 100 200 500 ; do
            echo "$input-$method-e$e-N$N-n$n.csv" 
            java -Xms3G -Xmx6G -cp $JAVA_CLASSPATH  de.tudarmstadt.lt.wsi.WSI \
                -clustering $method \
                -N $N \
                -n $n \
                -e 0 \
                -in $input \
                -out $input-$method-e$e-N$N-n$n.csv > $input-$method-e$e-N$N-n$n.log
        done
    done
done
