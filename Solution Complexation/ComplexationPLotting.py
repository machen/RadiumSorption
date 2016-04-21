# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 16:25:09 2016

@author: Michael
"""

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
import re

filePath = 'SrSolComplCO2.xlsx'
dataFile = pd.read_excel(filePath)
prob = re.compile('Problem')
run = re.compile('Run')
for name in dataFile.columns:
    if prob.match(name):
        continue            
    if run.match(name):
        break
    if name == 'pH':
        pH = dataFile.ix[:,'pH']
        