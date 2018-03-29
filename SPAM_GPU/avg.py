#! /usr/env/python

import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])

print(df[125:-25].mean()[0])
