dir=/mnt10/verbs/data/culwg/release/
for t in 10 20 ; do
	for n in 20 50 200 ; do
		f="${dir}/senses-culwg-coarse-t${t}-cw-e0-N200-n${n}-minsize5.csv.gz"
		echo "t=$t, n=$n, pcz=$f"
		python -u run_verbsim_par.py $f #> $f.unit-words.results
	done 
done
