rm powerlog.csv
while true; do
    sleep 0.04
    nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits >> powerlog.csv
done

