norm="sum"
max_words=20
weight_type="score"

for dim in 1000 2000 5000  ; do
    for t in 10 20 ; do
        for n in 20 50 200 ; do
            echo "============================================="
            echo "n=$n, t=$t"
            python build_sense_vectors.py \
                /mnt10/verbs/data/culwg/release/senses-culwg-coarse-t${t}-cw-e0-N200-n${n}-minsize5.csv.gz \
                /mnt10/verbs/data/culwg/release/lmi-culwg-coarse.csv.gz \
                --sparse \
                --max_dim $dim \
                --norm_type $norm \
                --max_words $max_words \
                --weight_type $weight_type
            gzip /mnt10/verbs/data/culwg/release/senses-culwg-coarse-t${t}-cw-e0-N200-n${n}-minsize5.csv.gz-${dim}-${norm}-${weight_type}-${max_words}.vectors.csv 
        done
    done
done
