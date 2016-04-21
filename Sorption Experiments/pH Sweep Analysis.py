# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:15:58 2015

@author: Michael
"""

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import linregress

saveData = False

#load Data
dataFileName = "pH Sweep FHY"
dataFilePath = "FHY_07_27_2015/"
rawData = pd.read_csv(dataFilePath+dataFileName+".csv",header=0)

#Plot Data
mineralMass = 4.00 #milligrams
dmineralMass = 0.01 #milligrams
Cw = rawData.ix[:,'Cw'].values
Cs = rawData.ix[:,'Cs'].values
dCw = rawData.ix[:,'sCw'].values
dCs = rawData.ix[:,'sCs'].values
pH = rawData.ix[:,'pH'].values
dpH = rawData.ix[:,'spH'].values
totAct = rawData.ix[:,'Total Activity'].values
fSolid = mineralMass*Cs/totAct
dfSolid = np.sqrt((dCs/Cs)**2+(dmineralMass/mineralMass)**2)*fSolid

f1 = plt.figure(1)
plt.clf()
perr = plt.errorbar(pH, fSolid, yerr = dfSolid, xerr = dpH, fmt='none')
p1, = plt.plot(pH, fSolid, "ob")
plt.title('pH Sweep Ferrihydrite, Total Bottle Activity = 100000 DPM')
plt.xlabel('pH')
plt.ylabel('Fraction Sorbed')
if saveData:
    plt.savefig(dataFilePath+"Sweep Plot Solid.pdf",dpi=1200)
