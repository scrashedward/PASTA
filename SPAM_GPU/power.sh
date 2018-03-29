#! /bin/bash

bash powerlog.sh &
timeout 800 $1 $2 $3 $4 $5 $6 $7 &
wait $!
ps au | grep "bash powerlog.sh" | grep -v grep | awk '{print $2}' | xargs kill
python avg.py powerlog.csv
rm powerlog.csv
