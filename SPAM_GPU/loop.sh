

for m in 2000 1750 1500 1400 1300 1250 1225 1200 1180 1170 1160 1150 1140 1130
do
	for s in 20 15 10
	do
		echo $m >> result/d100c10t5i0.1M
		echo $s >> result/d100c10t5i0.1M
		bash power.sh ./a.out data/data.ncust_100.slen_10.tlen_5.nitems_0.1 0.0$s -b 2048 -t 128 P -M $m
		mv memLog.csv $s$m.csv
	done
done
