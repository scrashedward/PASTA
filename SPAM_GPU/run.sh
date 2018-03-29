#! /bin/bash

c=10
t=5
i=1

for d in 200
do
	for bnum in 1024 2048 4096 8192 12288
	do
		for tnum in 64 128 256 512 1024
		do
			for sup in 02
			do
				for algo in zero
				do
					echo $sup >> result/${algo}.d${d}c${c}t${t}i${i}s${sup}
					echo $bnum >> result/${algo}.d${d}c${c}t${t}i${i}s${sup}
					echo $tnum >> result/${algo}.d${d}c${c}t${t}i${i}s${sup}
					bash power.sh ./${algo}.out data/data.ncust_${d}.slen_${c}.tlen_${t}.nitems_${i} 0.0$sup -b $bnum -t $tnum >> result/${algo}.d${d}c${c}t${t}i${i}s${sup}
				done
			done
		done
	done
done
