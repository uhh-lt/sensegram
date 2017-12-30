
for w in score rank ones ; do 
    for pcz in w2v jbt ; do
        for m in 999 ; do  # 5 10 20 ; do 
            python build_sense_vectors.py \
                /mnt10/verbs/${pcz}.csv.gz \
                /mnt10/verbs/all.norm-sz500-w10-cb0-it3-min5.w2v \
                --max_words $m \
                --weight_type $w
        done
    done
done 
